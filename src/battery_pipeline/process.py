"""Stage 2: per-cell pass over the lake -> cycles, QC flags, dQ/dV curves, features.

Each cell's time series is read once; per-cycle metrics, cycle QC flags,
cellpy dQ/dV curves for a few reference cycles and the early-cycle features
are all computed from that one read, so memory stays bounded by one cell.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from battery_pipeline import features as F
from battery_pipeline import quality, telemetry
from battery_pipeline.metrics import cycle_metrics, discharge_curve, dqdv
from battery_pipeline.store import Lake

#: Cycles whose dQ/dV curve is kept (10 and 100 feed the ICA features).
DQDV_CYCLES = (10, 100, 200, 500, 1000)


def _peak(curve: pd.DataFrame) -> tuple[float, float]:
    # main LFP discharge plateau sits between 3.0 and 3.4 V
    c = curve[(curve["voltage"] > 3.0) & (curve["voltage"] < 3.4)]
    if c.empty:
        return float("nan"), float("nan")
    i = c["dqdv"].idxmax()
    return float(c.at[i, "dqdv"]), float(c.at[i, "voltage"])


def process_cell(root: str, cell_id: str) -> dict:
    lake = Lake(Path(root))
    ts = lake.read_timeseries(cell_id)
    vendor = pd.read_parquet(lake.table("vendor_summary"), filters=[("cell_id", "==", cell_id)])
    cyc = cycle_metrics(ts, vendor)
    flags = quality.cycle_flags(ts, cyc, quality.rules_for(cell_id))

    curves, peaks = [], []
    for k in DQDV_CYCLES:
        one = ts[ts["cycle_index"] == k]
        if one.empty:
            continue
        v, q = discharge_curve(one)
        curve = dqdv(v, q)
        if curve.empty:
            continue
        curves.append(curve.assign(cell_id=cell_id, cycle_index=k))
        h, vp = _peak(curve)
        peaks.append({"cycle_index": k, "peak_dqdv": h, "peak_voltage": vp})

    feats: dict = {}
    feats_raw: dict = {}
    curves_lin = lake.read_curves(cell_id)
    n_cycles = int(cyc["cycle_index"].max()) if len(cyc) else 0
    if n_cycles >= 100 and len(curves_lin):
        qd = np.stack(curves_lin["qdlin"].to_numpy())
        td = np.stack(curves_lin["tdlin"].to_numpy())
        good = set(flags.loc[~flags["bad_cycle"], "cycle_index"].astype(int))
        feats = F.cell_features(qd, td, vendor, cyc, good_cycles=good)
        feats_raw = F.cell_features(qd, td, vendor, cyc, good_cycles=None)
        ica = F.ica_features(pd.DataFrame(peaks)) if peaks else {}
        feats.update(ica)
        feats_raw.update(ica)
    feats["cell_id"] = cell_id
    feats_raw["cell_id"] = cell_id
    ambient = vendor["ambient_c"].mean() if "ambient_c" in vendor.columns else np.nan
    return {
        "cell_id": cell_id,
        "rows": len(ts),
        "cycles": cyc,
        "flags": flags,
        "dqdv": pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
        "peaks": pd.DataFrame(peaks).assign(cell_id=cell_id) if peaks else pd.DataFrame(),
        "features": feats,
        "features_raw": feats_raw,
        "ambient_c": float(ambient),
    }


def run(lake: Lake, workers: int = 4, log: Callable[[str], None] = print) -> dict[str, int]:
    cell_ids = lake.cell_ids()
    out: list[dict] = []
    with telemetry.stage("process"):
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(process_cell, str(lake.root), c) for c in cell_ids]
                for i, fut in enumerate(futures, 1):
                    out.append(fut.result())
                    if i % 20 == 0 or i == len(futures):
                        log(f"  processed {i}/{len(futures)} cells")
        else:
            out = [process_cell(str(lake.root), c) for c in cell_ids]
        for r in out:
            telemetry.ROWS.labels("process").inc(r["rows"])
            telemetry.CYCLES.labels("process").inc(len(r["cycles"]))
        telemetry.CELLS.labels("process").inc(len(out))

        cycles = pd.concat([r["cycles"] for r in out], ignore_index=True)
        flags = pd.concat([r["flags"] for r in out], ignore_index=True)
        lake.write_table("cycles", cycles)
        lake.write_table("qc_cycle_flags", flags)
        dq = [r["dqdv"] for r in out if len(r["dqdv"])]
        if dq:
            lake.write_table("dqdv_curves", pd.concat(dq, ignore_index=True))
        pk = [r["peaks"] for r in out if len(r["peaks"])]
        if pk:
            lake.write_table("dqdv_peaks", pd.concat(pk, ignore_index=True))
        for key, table in (("features", "features"), ("features_raw", "features_noqc")):
            feats = pd.DataFrame([r[key] for r in out])
            feats = feats[["cell_id"] + [c for c in feats.columns if c != "cell_id"]]
            lake.write_table(table, feats)

        ind = quality.cell_indicators(cycles, flags)
        ind["source"] = np.where(ind["cell_id"].str.startswith("nasa_"), "nasa", "matr")
        ind["ambient_c"] = ind["cell_id"].map({r["cell_id"]: r["ambient_c"] for r in out})
        covariate = {"matr": "charge_time_mean_min", "nasa": "ambient_c"}
        flagged = [quality.cell_flags(g, covariate=covariate[src]) for src, g in ind.groupby("source")]
        lake.write_table("qc_cell_flags", pd.concat(flagged, ignore_index=True))
    return {"cells": len(out), "cycles": len(cycles), "rows": int(sum(r["rows"] for r in out))}
