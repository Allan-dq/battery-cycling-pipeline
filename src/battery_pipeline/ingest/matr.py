"""Reader for the Severson/Attia et al. (2019) MATR batch files.

The files are MATLAB v7.3 (HDF5) structs: ``batch`` holds one object
reference per cell for ``summary``, ``cycles``, ``cycle_life``,
``policy_readable``, ``barcode`` and ``channel_id``. ``cycles`` holds, per
cycle, references to the raw arrays ``t`` (minutes), ``I`` (C-rate, + = charge;
multiplied by the 1.1 Ah nominal capacity to get amperes),
``V`` (V), ``T`` (°C, can thermocouple), ``Qc``/``Qd`` (Ah, cumulative within
the cycle) and the dataset's derived ``Qdlin``/``Tdlin`` (discharge capacity
and temperature interpolated on the 1000-point voltage grid ``Vdlin``).

Cell ids follow the authors' loading code: ``b<batch>c<index>``. Cycle 1 is an
empty placeholder in every cell (two zero samples, zero summary capacity).
``barcode``/``channel_id`` are MATLAB ``string`` objects that h5py cannot
decode, so they are left empty.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from battery_pipeline.steps import label_steps

NOMINAL_CAPACITY_AH = 1.1
EOL_CAPACITY_AH = 0.88  # 80% of nominal: the paper's end-of-life definition

#: Batch label per file stem prefix (the authors' order: batch 1, 2, 3).
BATCH_OF_FILE = {"2017-05-12": "b1", "2017-06-30": "b2", "2018-04-12": "b3"}

#: Five batch-1 cells were moved to batch 2 mid-test and continue there.
#: (batch-2 id -> batch-1 id), from the authors' published loading code.
CONTINUES = {"b2c7": "b1c0", "b2c8": "b1c1", "b2c9": "b1c2", "b2c15": "b1c3", "b2c16": "b1c4"}
#: Cycles added to the batch-1 reported cycle life for those cells (same source).
CONTINUATION_EXTRA_LIFE = {"b1c0": 662, "b1c1": 981, "b1c2": 1060, "b1c3": 208, "b1c4": 482}


@dataclass
class MatrCell:
    cell_id: str
    batch: str
    charge_policy: str
    barcode: str
    channel: str
    reported_cycle_life: float
    timeseries: pd.DataFrame  # canonical TIMESERIES columns
    summary: pd.DataFrame  # dataset per-cycle summary, canonical names
    qdlin: np.ndarray  # (n_cycles, n_voltage) discharge capacity on the voltage grid
    tdlin: np.ndarray  # (n_cycles, n_voltage) temperature on the voltage grid
    cycle_numbers: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))


def batch_of(path: Path) -> str:
    for prefix, label in BATCH_OF_FILE.items():
        if Path(path).name.startswith(prefix):
            return label
    match = re.search(r"(b\d+)", Path(path).stem)
    if match:  # fixtures: "matr_sample_b1.h5"
        return match.group(1)
    raise ValueError(f"cannot tell the batch of {path}")


def _deref(f: h5py.File, ref) -> h5py.Dataset | h5py.Group:
    return f[ref]


def _string(f: h5py.File, ref) -> str:
    """Decode a MATLAB ``char`` array; MATLAB ``string`` objects come back empty."""
    ds = f[ref]
    if ds.attrs.get("MATLAB_class", b"") not in (b"char", "char"):
        return ""
    return "".join(chr(c) for c in np.asarray(ds[()]).ravel() if c)


def _scalar(f: h5py.File, ref) -> float:
    return float(np.asarray(f[ref][()]).ravel()[0])


def _vector(f: h5py.File, ref) -> np.ndarray:
    return np.asarray(f[ref][()], dtype=np.float64).ravel()


def voltage_grid(path: Path) -> np.ndarray:
    """The 1000-point voltage grid the dataset interpolates ``Qdlin`` on."""
    with h5py.File(path, "r") as f:
        batch = f["batch"]
        if "Vdlin" in batch:
            return _vector(f, batch["Vdlin"][0, 0])
    return np.linspace(3.6, 2.0, 1000)


def read_cells(path: Path, cell_ids: set[str] | None = None) -> Iterator[MatrCell]:
    """Yield every cell in one MATR batch file, one at a time (bounded memory)."""
    path = Path(path)
    label = batch_of(path)
    with h5py.File(path, "r") as f:
        batch = f["batch"]
        n_cells = batch["summary"].shape[0]
        # trimmed fixtures keep each cell's position in the full file
        index = batch["cell_index"][()].ravel().astype(int) if "cell_index" in batch else range(n_cells)
        for i in range(n_cells):
            cell_id = f"{label}c{index[i]}"
            if cell_ids is not None and cell_id not in cell_ids:
                continue
            yield _read_cell(f, batch, i, cell_id, label)


def _read_cell(f: h5py.File, batch: h5py.Group, i: int, cell_id: str, label: str) -> MatrCell:
    summary_grp = f[batch["summary"][i, 0]]
    summary = pd.DataFrame(
        {
            "cycle_index": np.asarray(summary_grp["cycle"][()]).ravel().astype(np.int32),
            "ir_ohm": np.asarray(summary_grp["IR"][()]).ravel(),
            "vendor_charge_capacity_ah": np.asarray(summary_grp["QCharge"][()]).ravel(),
            "vendor_discharge_capacity_ah": np.asarray(summary_grp["QDischarge"][()]).ravel(),
            "vendor_temperature_mean_c": np.asarray(summary_grp["Tavg"][()]).ravel(),
            "vendor_temperature_min_c": np.asarray(summary_grp["Tmin"][()]).ravel(),
            "vendor_temperature_max_c": np.asarray(summary_grp["Tmax"][()]).ravel(),
            "vendor_charge_time_min": np.asarray(summary_grp["chargetime"][()]).ravel(),
        }
    )

    cycles = f[batch["cycles"][i, 0]]
    n_cycles = cycles["I"].shape[0]
    frames, qdlin, tdlin = [], [], []
    offset_s = 0.0
    for j in range(n_cycles):
        t_min = _vector(f, cycles["t"][j, 0])
        current = _vector(f, cycles["I"][j, 0]) * NOMINAL_CAPACITY_AH  # C-rate -> A
        n = len(t_min)
        cycle_time_s = (t_min - t_min[0]) * 60.0 if n else t_min
        frame = pd.DataFrame(
            {
                "cell_id": cell_id,
                "cycle_index": np.full(n, j + 1, dtype=np.int32),
                "test_time_s": offset_s + cycle_time_s,
                "cycle_time_s": cycle_time_s,
                "current_a": current,
                "voltage_v": _vector(f, cycles["V"][j, 0]),
                "temperature_c": _vector(f, cycles["T"][j, 0]),
                "charge_capacity_ah": _vector(f, cycles["Qc"][j, 0]),
                "discharge_capacity_ah": _vector(f, cycles["Qd"][j, 0]),
            }
        )
        if n:
            step_type, step_index = label_steps(current)
            frame["step_type"] = step_type
            frame["step_index"] = step_index
            offset_s = float(frame["test_time_s"].iloc[-1])
        else:
            frame["step_type"] = pd.Series(dtype=str)
            frame["step_index"] = pd.Series(dtype=np.int16)
        frames.append(frame)
        qdlin.append(_vector(f, cycles["Qdlin"][j, 0]))
        tdlin.append(_vector(f, cycles["Tdlin"][j, 0]))

    width = max((len(q) for q in qdlin), default=0)
    qd = np.full((n_cycles, width), np.nan, dtype=np.float32)
    td = np.full((n_cycles, width), np.nan, dtype=np.float32)
    for j, (q, t) in enumerate(zip(qdlin, tdlin, strict=True)):
        qd[j, : len(q)] = q
        td[j, : len(t)] = t

    return MatrCell(
        cell_id=cell_id,
        batch=label,
        charge_policy=_string(f, batch["policy_readable"][i, 0]),
        barcode="",
        channel="",
        reported_cycle_life=_scalar(f, batch["cycle_life"][i, 0]),
        timeseries=pd.concat(frames, ignore_index=True),
        summary=summary,
        qdlin=qd,
        tdlin=td,
        cycle_numbers=np.arange(1, n_cycles + 1, dtype=np.int32),
    )


def cycle_life_from_summary(discharge_capacity_ah: np.ndarray, threshold: float = EOL_CAPACITY_AH) -> float:
    """First 1-based cycle whose discharge capacity is below ``threshold``.

    Returns NaN when the cell never crosses it. Cycle 1 is skipped: its
    capacity is often missing or zero in these files.
    """
    q = np.asarray(discharge_capacity_ah, dtype=float)
    below = np.flatnonzero((q < threshold) & (np.arange(len(q)) >= 1) & (q > 0))
    return float(below[0] + 1) if len(below) else float("nan")
