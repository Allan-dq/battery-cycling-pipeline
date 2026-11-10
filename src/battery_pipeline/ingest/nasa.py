"""Adapter for the NASA Ames PCoE "Battery Data Set" (Saha & Goebel, 2007).

Each ``B00xx.mat`` (MATLAB v5) holds a ``cycle`` struct array of
operations, each ``charge``, ``discharge`` or ``impedance`` with its own
time vector (s from the operation start) and a start timestamp (MATLAB date
vector). Current is positive while charging, negative while discharging,
the same convention as the canonical schema.

A canonical cycle here is one charge plus the discharge(s) that follow it,
up to the next charge. Impedance sweeps are not time series and are skipped.
NASA reports the discharge capacity of each discharge; that is kept as the
vendor value. The README does not state the cell chemistry, so none is
claimed; the rated capacity is 2 Ah.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from battery_pipeline.steps import label_steps

NOMINAL_CAPACITY_AH = 2.0
EOL_CAPACITY_AH = 1.4  # NASA's end-of-life criterion: 30% fade from 2 Ah


@dataclass
class NasaCell:
    cell_id: str
    ambient_c: str
    protocol: str
    timeseries: pd.DataFrame
    summary: pd.DataFrame


def _first(value) -> float:
    """First element as float; NaN when the field is missing or empty."""
    arr = np.ravel(np.asarray(value, dtype=float)) if value is not None else np.empty(0)
    return float(arr[0]) if arr.size else float("nan")


def _datevec(v) -> datetime:
    y, mo, d, h, mi, s = (float(x) for x in np.ravel(v)[:6])
    return datetime(int(y), int(mo), int(d), int(h), int(mi)) + timedelta(seconds=s)


def _protocol(readme: Path | None, number: int) -> str:
    if readme is None or not readme.exists():
        return ""
    text = readme.read_text(errors="ignore")
    first = text.split("Files:")[0].replace("\n", " ")
    return re.sub(r"\s+", " ", first).strip()[:400]


def read_cell(path: Path) -> NasaCell:
    path = Path(path)
    m = sio.loadmat(path, simplify_cells=True)
    key = next(k for k in m if not k.startswith("__"))
    ops = m[key]["cycle"]
    if isinstance(ops, dict):
        ops = [ops]
    t0 = None
    cycle = 0
    frames, summary = [], []
    ambients = set()
    for op in ops:
        kind = op["type"]
        if kind not in ("charge", "discharge"):
            continue
        d = op["data"]
        start = _datevec(op["time"])
        t0 = t0 or start
        if kind == "charge" or cycle == 0:
            cycle += 1
            cycle_start = start
        ambients.add(int(op["ambient_temperature"]))
        t = np.asarray(d["Time"], dtype=float).ravel()
        offset = (start - cycle_start).total_seconds()
        n = len(t)
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": f"nasa_{key}",
                    "cycle_index": np.full(n, cycle, dtype=np.int32),
                    "test_time_s": (start - t0).total_seconds() + t,
                    "cycle_time_s": offset + t,
                    "current_a": np.asarray(d["Current_measured"], dtype=float).ravel(),
                    "voltage_v": np.asarray(d["Voltage_measured"], dtype=float).ravel(),
                    "temperature_c": np.asarray(d["Temperature_measured"], dtype=float).ravel(),
                    "charge_capacity_ah": np.nan,
                    "discharge_capacity_ah": np.nan,
                }
            )
        )
        if kind == "discharge":
            summary.append(
                {
                    "cycle_index": cycle,
                    "vendor_discharge_capacity_ah": _first(d.get("Capacity")),
                    "ambient_c": float(op["ambient_temperature"]),
                }
            )
    ts = pd.concat(frames, ignore_index=True).sort_values(["cycle_index", "cycle_time_s"], kind="stable")
    ts = ts.reset_index(drop=True)
    # cumulative within-cycle capacities by coulomb counting (NASA does not log them)
    for col, sign in (("charge_capacity_ah", 1.0), ("discharge_capacity_ah", -1.0)):
        i = np.clip(sign * ts["current_a"].to_numpy(), 0, None)
        dt = ts.groupby("cycle_index")["cycle_time_s"].diff().fillna(0).to_numpy()
        ts[col] = pd.Series(i * dt / 3600.0).groupby(ts["cycle_index"]).cumsum().to_numpy()
    step_type = np.empty(len(ts), dtype=object)
    step_index = np.zeros(len(ts), dtype=np.int16)
    for _, idx in ts.groupby("cycle_index").indices.items():
        st, si = label_steps(ts["current_a"].to_numpy()[idx])
        step_type[idx], step_index[idx] = st, si
    ts["step_type"], ts["step_index"] = step_type, step_index

    s = pd.DataFrame(summary)
    if len(s):
        s = s.groupby("cycle_index", as_index=False).agg(
            vendor_discharge_capacity_ah=("vendor_discharge_capacity_ah", lambda x: x.sum(min_count=1)),
            ambient_c=("ambient_c", "first"),
        )
    readme = next(iter(sorted(path.parent.glob("README*.txt"))), None)
    return NasaCell(
        cell_id=f"nasa_{key}",
        ambient_c="/".join(str(a) for a in sorted(ambients)),
        protocol=_protocol(readme, int(key[1:])),
        timeseries=ts,
        summary=s,
    )


def find_files(raw: Path) -> list[Path]:
    """Every B00xx.mat under ``raw``; later archives win when a cell appears twice."""
    by_cell: dict[str, Path] = {}
    for p in sorted(Path(raw).rglob("B00*.mat")):
        by_cell[p.stem] = p  # "3. ..._25-44" sorts after "2. ..._P1" (part 1)
    return [by_cell[k] for k in sorted(by_cell)]


def read_cells(raw: Path) -> Iterator[NasaCell]:
    for p in find_files(raw):
        yield read_cell(p)
