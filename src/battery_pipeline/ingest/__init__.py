"""Raw sources -> canonical Parquet lake."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from battery_pipeline import telemetry
from battery_pipeline.ingest import matr, nasa
from battery_pipeline.schema import CELLS, TIMESERIES, conform
from battery_pipeline.store import Lake


def _upsert(lake: Lake, name: str, df: pd.DataFrame, cell_ids: set[str]) -> None:
    """Replace this run's cells in a shared table, keep every other source's rows."""
    path = lake.table(name)
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~old["cell_id"].isin(cell_ids)]
        df = pd.concat([old, df], ignore_index=True) if len(old) else df
    lake.write_table(name, df)


@dataclass
class IngestReport:
    files: list[str] = field(default_factory=list)
    cells: int = 0
    cycles: int = 0
    rows: int = 0
    merged_continuations: list[str] = field(default_factory=list)


def _batch_order(path: Path) -> str:
    return matr.batch_of(path)


def ingest_matr(
    raw_files: Iterable[Path],
    lake: Lake,
    cell_ids: set[str] | None = None,
    log: Callable[[str], None] = print,
) -> IngestReport:
    """Read MATR batch files into the lake, merging the five continued cells.

    Files are processed in batch order so that a batch-2 continuation can be
    appended to its batch-1 cell (cycle numbers and test time shifted to
    follow on). Returns counts for the run log.
    """
    report = IngestReport()
    registry: dict[str, dict] = {}
    summaries: list[pd.DataFrame] = []
    with telemetry.stage("ingest"):
        for path in sorted(map(Path, raw_files), key=_batch_order):
            report.files.append(path.name)
            log(f"ingest {path.name}")
            for cell in matr.read_cells(path, cell_ids):
                target = matr.CONTINUES.get(cell.cell_id)
                ts, summary = cell.timeseries, cell.summary.copy()
                if target and target in registry:
                    prev = registry[target]
                    ts = ts.assign(
                        cell_id=target,
                        cycle_index=ts["cycle_index"] + prev["n_cycles"],
                        test_time_s=ts["test_time_s"] + prev["last_time_s"],
                    )
                    summary["cycle_index"] = summary["cycle_index"] + prev["n_cycles"]
                    report.merged_continuations.append(f"{cell.cell_id}->{target}")
                    cell_id = target
                else:
                    cell_id = cell.cell_id
                    prev = None

                lake.write_partition("timeseries", cell_id, cell.batch, conform(ts, TIMESERIES))
                cycle_numbers = cell.cycle_numbers + (prev["n_cycles"] if prev else 0)
                curves = pa.table(
                    {
                        "cycle_index": pa.array(cycle_numbers, pa.int32()),
                        "qdlin": pa.array(list(cell.qdlin), pa.list_(pa.float32())),
                        "tdlin": pa.array(list(cell.tdlin), pa.list_(pa.float32())),
                    }
                )
                lake.write_partition("curves", cell_id, cell.batch, curves)
                summaries.append(summary.assign(cell_id=cell_id))

                n_cycles = len(cell.cycle_numbers)
                last_time = float(ts["test_time_s"].max()) if len(ts) else 0.0
                if prev:
                    prev["n_cycles"] += n_cycles
                    prev["last_time_s"] = last_time
                    prev["reported_cycle_life"] += matr.CONTINUATION_EXTRA_LIFE.get(target, 0)
                else:
                    registry[cell_id] = {
                        "cell_id": cell_id,
                        "source": "matr",
                        "batch": cell.batch,
                        "chemistry": "LFP/graphite (A123 APR18650M1A)",
                        "barcode": cell.barcode,
                        "channel": cell.channel,
                        "charge_policy": cell.charge_policy,
                        "nominal_capacity_ah": matr.NOMINAL_CAPACITY_AH,
                        "n_cycles": n_cycles,
                        "reported_cycle_life": cell.reported_cycle_life,
                        "last_time_s": last_time,
                    }
                report.rows += len(ts)
                report.cycles += n_cycles
                telemetry.ROWS.labels("ingest").inc(len(ts))
                telemetry.CYCLES.labels("ingest").inc(n_cycles)
            log(f"  {len(registry)} cells so far, {report.rows:,} rows")

        vendor = pd.concat(summaries, ignore_index=True)
        vendor = vendor[["cell_id"] + [c for c in vendor.columns if c != "cell_id"]]
        _upsert(lake, "vendor_summary", vendor, set(registry))

        cells = pd.DataFrame(registry.values()).drop(columns=["last_time_s"])
        life = {
            cid: matr.cycle_life_from_summary(g.sort_values("cycle_index")["vendor_discharge_capacity_ah"].to_numpy())
            for cid, g in vendor.groupby("cell_id")
        }
        cells["cycle_life"] = cells["cell_id"].map(life).astype(float)
        cells["n_cycles"] = cells["n_cycles"].astype(np.int32)
        _upsert(lake, "cells", conform(cells, CELLS).to_pandas(), set(registry))
        report.cells = len(cells)
        telemetry.CELLS.labels("ingest").inc(len(cells))
    return report


def ingest_nasa(raw: Path, lake: Lake, log: Callable[[str], None] = print) -> IngestReport:
    """Read the NASA PCoE B00xx.mat files into the lake (no Q(V) curves)."""
    report = IngestReport()
    rows, summaries = [], []
    with telemetry.stage("ingest_nasa"):
        for cell in nasa.read_cells(raw):
            ts = cell.timeseries
            lake.write_partition("timeseries", cell.cell_id, "nasa", conform(ts, TIMESERIES))
            summaries.append(cell.summary.assign(cell_id=cell.cell_id))
            n_cycles = int(ts["cycle_index"].max())
            q = cell.summary.sort_values("cycle_index")["vendor_discharge_capacity_ah"].to_numpy()
            below = np.flatnonzero(q < nasa.EOL_CAPACITY_AH)
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "source": "nasa",
                    "batch": "nasa",
                    "chemistry": "Li-ion 18650, 2 Ah rated (chemistry not stated by NASA)",
                    "barcode": "",
                    "channel": "",
                    "charge_policy": f"ambient {cell.ambient_c} C; {cell.protocol[:200]}",
                    "nominal_capacity_ah": nasa.NOMINAL_CAPACITY_AH,
                    "n_cycles": n_cycles,
                    "reported_cycle_life": np.nan,
                    "cycle_life": float(cell.summary["cycle_index"].iloc[below[0]]) if len(below) else np.nan,
                }
            )
            report.rows += len(ts)
            report.cycles += n_cycles
            telemetry.ROWS.labels("ingest_nasa").inc(len(ts))
            telemetry.CYCLES.labels("ingest_nasa").inc(n_cycles)
        report.cells = len(rows)
        log(f"  {report.cells} NASA cells, {report.rows:,} rows")
        cells = pd.DataFrame(rows)
        _upsert(lake, "cells", conform(cells, CELLS).to_pandas(), set(cells["cell_id"]))
        vendor = pd.concat(summaries, ignore_index=True)
        vendor = vendor[["cell_id"] + [c for c in vendor.columns if c != "cell_id"]]
        _upsert(lake, "vendor_summary", vendor, set(cells["cell_id"]))
        telemetry.CELLS.labels("ingest_nasa").inc(len(rows))
    return report
