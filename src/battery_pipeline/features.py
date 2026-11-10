"""Early-cycle features from Severson, Attia et al., Nature Energy 4, 383 (2019).

All features use cycles 2-100 only (cycle 1 is an empty placeholder in the
MATR files). Names follow the paper's feature table; implementation choices
the paper leaves to its supplement are stated next to each feature:

* ΔQ statistics use ΔQ_{100-10}(V) on the dataset's 1000-point grid
  (3.5 V -> 2.0 V) and are log10(|x|), as in the "variance" model.
* variance uses ddof=1 and kurtosis is Pearson (not excess), matching MATLAB
  defaults, which the authors used.
* capacity, charge time, internal resistance and min/max temperature come
  from the dataset's per-cycle summary; the temperature integral comes from
  the raw can-temperature time series (°C·min summed over cycles 2-100).
* ``good_cycles`` (from QC) drops broken cycles before any statistic is
  taken: a cycle that was interrupted and resumed, or whose summary capacity
  disagrees with the raw current, would otherwise set e.g. the "max
  capacity" feature. If cycle 10 or 100 itself is bad, the nearest good cycle
  within ±3 is used for ΔQ(V). ``good_cycles=None`` reproduces the raw
  (uncleaned) features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from battery_pipeline.metrics import delta_q

#: The three models in the paper's results table.
VARIANCE_FEATURES = ["dq_var"]
DISCHARGE_FEATURES = ["dq_min", "dq_var", "dq_skew", "dq_kurt", "qd2", "qd_max_minus_2"]
FULL_FEATURES = [
    "dq_min",
    "dq_var",
    "fade_slope_2_100",
    "fade_intercept_2_100",
    "qd2",
    "charge_time_first5",
    "t_integral_2_100",
    "ir_min_2_100",
    "ir_diff_100_2",
]
#: All 20 candidate features of the paper's feature table.
CANDIDATE_FEATURES = [
    "dq_min",
    "dq_mean",
    "dq_var",
    "dq_skew",
    "dq_kurt",
    "dq_2v",
    "fade_slope_2_100",
    "fade_intercept_2_100",
    "fade_slope_91_100",
    "fade_intercept_91_100",
    "qd2",
    "qd_max_minus_2",
    "qd100",
    "charge_time_first5",
    "t_max_2_100",
    "t_min_2_100",
    "t_integral_2_100",
    "ir2",
    "ir_min_2_100",
    "ir_diff_100_2",
]
#: Extra temperature features from ΔT_{100-10}(V) (dataset's Tdlin), tried as an improvement.
TEMPERATURE_EXTRA = ["dt_var", "dt_mean", "tmax_slope_2_100"]
#: cellpy incremental-capacity (dQ/dV) features, tried as an improvement.
ICA_EXTRA = ["ica_peak_ratio_100_10", "ica_peak_shift_mv_100_10"]


def _log_abs(x: float) -> float:
    return float(np.log10(np.abs(x))) if np.isfinite(x) and x != 0 else float("nan")


def _fit(cycles: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(q) & np.isfinite(cycles)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(cycles[ok], q[ok], 1)
    return float(slope), float(intercept)


def _nearest_good(cycle: int, good: set[int] | None, n_cycles: int, reach: int = 3) -> int:
    if good is None or cycle in good:
        return cycle
    for d in range(1, reach + 1):
        for c in (cycle - d, cycle + d):
            if 2 <= c <= n_cycles and c in good:
                return c
    return cycle


def cell_features(
    qdlin: np.ndarray,
    tdlin: np.ndarray | None,
    summary: pd.DataFrame,
    cycle_metrics: pd.DataFrame,
    good_cycles: set[int] | None = None,
) -> dict[str, float]:
    """Features for one cell.

    ``qdlin``/``tdlin`` are (n_cycles, n_voltage) with row k-1 = cycle k.
    ``summary`` is the dataset per-cycle summary (vendor_* columns, ir_ohm)
    and ``cycle_metrics`` our per-cycle metrics; both indexed by cycle_index.
    """
    s = summary.set_index("cycle_index").sort_index()
    m = cycle_metrics.set_index("cycle_index").sort_index()
    if good_cycles is not None:
        s = s[s.index.isin(good_cycles)]
        m = m[m.index.isin(good_cycles)]
    early = s.loc[2:100]
    q = early["vendor_discharge_capacity_ah"].to_numpy(float)
    cyc = early.index.to_numpy(float)

    c100 = _nearest_good(100, good_cycles, len(qdlin))
    c10 = _nearest_good(10, good_cycles, len(qdlin))
    dq = delta_q(qdlin, c100, c10)
    feats: dict[str, float] = {
        "dq_min": _log_abs(np.nanmin(dq)),
        "dq_mean": _log_abs(np.nanmean(dq)),
        "dq_var": _log_abs(np.nanvar(dq, ddof=1)),
        "dq_skew": _log_abs(stats.skew(dq, nan_policy="omit")),
        "dq_kurt": _log_abs(stats.kurtosis(dq, fisher=False, nan_policy="omit")),
        "dq_2v": _log_abs(dq[np.isfinite(dq)][-1]) if np.isfinite(dq).any() else float("nan"),
    }
    feats["fade_slope_2_100"], feats["fade_intercept_2_100"] = _fit(cyc, q)
    late = (cyc >= 91) & (cyc <= 100)
    feats["fade_slope_91_100"], feats["fade_intercept_91_100"] = _fit(cyc[late], q[late])
    n = int(s.index.max()) if len(s) else 0
    c2, c100s = _nearest_good(2, good_cycles, n), _nearest_good(100, good_cycles, n)

    def at(cycle: int, col: str) -> float:
        return float(s.at[cycle, col]) if cycle in s.index else float("nan")

    feats["qd2"] = at(c2, "vendor_discharge_capacity_ah")
    feats["qd_max_minus_2"] = float(np.nanmax(q) - feats["qd2"])
    feats["qd100"] = at(c100s, "vendor_discharge_capacity_ah")
    feats["charge_time_first5"] = float(s.loc[2:6, "vendor_charge_time_min"].mean())
    feats["t_max_2_100"] = float(early["vendor_temperature_max_c"].max())
    feats["t_min_2_100"] = float(early["vendor_temperature_min_c"].min())
    # mean per-cycle integral x 99 cycles = the sum when no cycle is dropped
    feats["t_integral_2_100"] = float(m.loc[2:100, "temperature_integral_c_min"].mean() * 99)
    ir = early["ir_ohm"].where(early["ir_ohm"] > 0)
    feats["ir2"] = float(ir.get(c2, np.nan))
    feats["ir_min_2_100"] = float(ir.min())
    feats["ir_diff_100_2"] = float(ir.get(c100s, np.nan) - ir.get(c2, np.nan))

    if tdlin is not None:
        dt = delta_q(tdlin, c100, c10)
        feats["dt_var"] = _log_abs(np.nanvar(dt, ddof=1))
        feats["dt_mean"] = float(np.nanmean(dt))
    tmax = early["vendor_temperature_max_c"].to_numpy(float)
    feats["tmax_slope_2_100"] = _fit(cyc, tmax)[0]
    feats["n_bad_cycles_2_100"] = float(99 - len(early)) if good_cycles is not None else 0.0
    feats["dq_cycles"] = f"{c100}-{c10}"
    return feats


def ica_features(peaks: pd.DataFrame) -> dict[str, float]:
    """dQ/dV peak change between cycle 10 and cycle 100 (from cellpy ICA curves).

    ``peaks`` has one row per cycle with ``peak_dqdv`` and ``peak_voltage``.
    """
    p = peaks.set_index("cycle_index")
    if not {10, 100} <= set(p.index):
        return {k: float("nan") for k in ICA_EXTRA}
    return {
        "ica_peak_ratio_100_10": float(p.at[100, "peak_dqdv"] / p.at[10, "peak_dqdv"]),
        "ica_peak_shift_mv_100_10": float((p.at[100, "peak_voltage"] - p.at[10, "peak_voltage"]) * 1e3),
    }
