"""Per-cycle metrics from canonical time series, plus dQ/dV and ΔQ(V) curves.

Capacities and energies are recomputed by integrating the raw current and
power over time (coulomb counting), not copied from the cycler, so the same
code works for any source; the dataset's own numbers are kept next to them as
``vendor_*`` columns for cross-checking.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from battery_pipeline.schema import CYCLES

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed it


def _integral(y: np.ndarray, t: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(t)
    if ok.sum() < 2:
        return float("nan")
    return float(_trapz(y[ok], t[ok]))


def _dcir_proxy(
    step_type: np.ndarray, current: np.ndarray, voltage: np.ndarray, min_step_a: float = 0.5
) -> float:
    """|ΔV/ΔI| across the first sample that enters a discharge step.

    It includes whatever polarisation builds up within one sampling
    interval, so it is a proxy for DC resistance, not a pulse measurement.
    """
    is_dis = step_type == "discharge"
    enter = np.flatnonzero(is_dis[1:] & ~is_dis[:-1]) + 1
    for k in enter:
        d_i = current[k - 1] - current[k]
        if d_i > min_step_a:
            return float((voltage[k - 1] - voltage[k]) / d_i)
    return float("nan")


def cycle_metrics(ts: pd.DataFrame, vendor_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (cell, cycle) with the ``CYCLES`` columns.

    ``ts`` is canonical time series for one or more cells. ``vendor_summary``
    (optional) carries dataset-reported values keyed by cell_id/cycle_index:
    ``vendor_charge_capacity_ah``, ``vendor_discharge_capacity_ah``, ``ir_ohm``.
    """
    rows = []
    for (cell_id, cycle), g in ts.groupby(["cell_id", "cycle_index"], sort=True):
        t = g["cycle_time_s"].to_numpy(float)
        current = g["current_a"].to_numpy(float)
        voltage = g["voltage_v"].to_numpy(float)
        temp = g["temperature_c"].to_numpy(float)
        steps = g["step_type"].to_numpy()
        order = np.argsort(t, kind="stable")
        t, current, voltage, temp, steps = t[order], current[order], voltage[order], temp[order], steps[order]
        dt = np.diff(t)

        q_ch = _integral(np.clip(current, 0, None), t) / 3600.0
        q_dis = _integral(np.clip(-current, 0, None), t) / 3600.0
        e_ch = _integral(np.clip(current, 0, None) * voltage, t) / 3600.0
        e_dis = _integral(np.clip(-current, 0, None) * voltage, t) / 3600.0
        charge_time_min = float(dt[steps[:-1] == "charge"].sum() / 60.0) if len(dt) else float("nan")
        has_temp = np.isfinite(temp).any()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rows.append(
                {
                    "cell_id": cell_id,
                    "cycle_index": int(cycle),
                    "n_samples": len(t),
                    "duration_s": float(t[-1] - t[0]) if len(t) else float("nan"),
                    "max_time_gap_s": float(dt.max()) if len(dt) else float("nan"),
                    "charge_capacity_ah": q_ch,
                    "discharge_capacity_ah": q_dis,
                    "coulombic_efficiency": q_dis / q_ch if q_ch and q_ch > 0 else float("nan"),
                    "charge_energy_wh": e_ch,
                    "discharge_energy_wh": e_dis,
                    "energy_efficiency": e_dis / e_ch if e_ch and e_ch > 0 else float("nan"),
                    "charge_time_min": charge_time_min,
                    "temperature_mean_c": float(np.nanmean(temp)) if has_temp else float("nan"),
                    "temperature_max_c": float(np.nanmax(temp)) if has_temp else float("nan"),
                    "temperature_min_c": float(np.nanmin(temp)) if has_temp else float("nan"),
                    "temperature_integral_c_min": _integral(temp, t / 60.0) if has_temp else float("nan"),
                    "dcir_proxy_ohm": _dcir_proxy(steps, current, voltage) if len(t) > 1 else float("nan"),
                }
            )
    out = pd.DataFrame(rows)
    vendor_cols = ["vendor_charge_capacity_ah", "vendor_discharge_capacity_ah", "ir_ohm"]
    if vendor_summary is not None and len(out):
        keep = ["cell_id", "cycle_index"] + [c for c in vendor_cols if c in vendor_summary.columns]
        out = out.merge(vendor_summary[keep], on=["cell_id", "cycle_index"], how="left")
    for c in vendor_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[CYCLES.names] if len(out) else pd.DataFrame(columns=CYCLES.names)


def delta_q(qdlin: np.ndarray, cycle_a: int, cycle_b: int) -> np.ndarray:
    """ΔQ_{a-b}(V) = Q_a(V) - Q_b(V) on the shared voltage grid (1-based cycles)."""
    return np.asarray(qdlin[cycle_a - 1], dtype=float) - np.asarray(qdlin[cycle_b - 1], dtype=float)


def discharge_curve(ts_cycle: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(voltage, discharged capacity) for the discharge part of one cycle."""
    d = ts_cycle[ts_cycle["step_type"] == "discharge"]
    return d["voltage_v"].to_numpy(float), d["discharge_capacity_ah"].to_numpy(float)


def dqdv(voltage: np.ndarray, capacity: np.ndarray, resolution_v: float = 0.005) -> pd.DataFrame:
    """Incremental capacity dQ/dV of one half-cycle, computed with cellpy's ICA.

    cellpy (``cellpy.utils.ica.dqdv``) interpolates the curve, smooths it and
    differentiates. Returned columns: ``voltage``, ``capacity``, ``dqdv``.
    Capacity here is the amount discharged so far, so dQ/dV is reported as a
    positive quantity for discharge.
    """
    from cellpy.utils import ica

    v = np.asarray(voltage, dtype=float)
    q = np.asarray(capacity, dtype=float)
    ok = np.isfinite(v) & np.isfinite(q)
    v, q = v[ok], q[ok]
    if len(v) < 10:
        return pd.DataFrame(columns=["voltage", "capacity", "dqdv"])
    frame = ica.dqdv((v, q), voltage_resolution=resolution_v)
    frame = frame[["voltage", "capacity", "dqdv"]].copy()
    frame.attrs = {}  # cellpy stores its IcaOptions here; not serialisable to Parquet
    frame["dqdv"] = frame["dqdv"].abs()
    return frame.reset_index(drop=True)
