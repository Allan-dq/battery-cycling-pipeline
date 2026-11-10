"""Data-quality checks (per cycle) and anomalous-cell detection (per cell).

The rules and thresholds are written down in ``docs/outlier_criteria.md``;
keep the two in sync. Thresholds come from the cell datasheet / test
protocol or from robust statistics, never from looking at which cells the
paper excluded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class CycleRules:
    min_samples: int = 10  # fewer samples = an empty/placeholder cycle
    max_gap_s: float = 60.0  # 12x the 5 s logging interval
    v_min: float = 1.9  # 0.1 V outside the 2.0-3.6 V protocol cutoffs
    v_max: float = 3.7
    max_abs_current_a: float = 11.0  # 10C; the fastest policy charges at 8C (8.8 A)
    t_min_c: float = 15.0  # chamber is held at 30 °C
    t_max_c: float = 60.0
    max_temp_step_c: float = 3.0  # |ΔT| between consecutive 5 s samples
    capacity_rel_tol: float = 0.02  # coulomb counting vs dataset-reported discharge capacity
    ce_min: float = 0.95
    ce_max: float = 1.05


#: Per-source limits. NASA cells are 4.2 V cells cycled at 4-44 °C ambient
#: (README cutoffs 2.0-2.7 V, loads up to 4 A), and a canonical NASA cycle
#: spans a charge, a rest/impedance sweep and a discharge, so long pauses
#: inside a cycle are normal there.
RULES_BY_SOURCE = {
    "matr": CycleRules(),
    "nasa": CycleRules(
        max_gap_s=4 * 3600.0, v_min=1.8, v_max=4.3, max_abs_current_a=5.0, t_min_c=-5.0, t_max_c=80.0
    ),
}


def rules_for(cell_id: str) -> CycleRules:
    return RULES_BY_SOURCE["nasa" if cell_id.startswith("nasa_") else "matr"]


@dataclass(frozen=True)
class CellRules:
    robust_z: float = 3.5  # Iglewicz & Hoaglin (1993) modified z-score cut-off
    early_first: int = 2
    early_last: int = 100
    temp_fault_frac: float = 0.05  # >5% of early cycles with a temperature fault
    iforest_contamination: float = 0.05
    iforest_seed: int = 0


#: Flags that make a cycle unusable for features and cell indicators. A long
#: pause alone (time_gap) does not: capacity and voltage curve are still valid.
BAD_CYCLE_FLAGS = ("empty_cycle", "time_not_monotonic", "capacity_mismatch", "ce_out_of_range")

CYCLE_FLAGS = (
    "empty_cycle",
    "duplicate_cycle",
    "time_gap",
    "time_not_monotonic",
    "voltage_out_of_range",
    "current_out_of_range",
    "temperature_missing",
    "temperature_out_of_range",
    "temperature_spike",
    "capacity_mismatch",
    "ce_out_of_range",
)


def cycle_flags(ts: pd.DataFrame, cycles: pd.DataFrame, rules: CycleRules = CycleRules()) -> pd.DataFrame:
    """Boolean QC flags per (cell, cycle). ``cycles`` is the metrics table for the same cells."""
    g = ts.groupby(["cell_id", "cycle_index"], sort=True)
    temp = ts["temperature_c"]
    t_diff = g["cycle_time_s"].diff()
    temp_step = g["temperature_c"].diff().abs()
    sample = pd.DataFrame(
        {
            "cell_id": ts["cell_id"],
            "cycle_index": ts["cycle_index"],
            "neg_dt": t_diff < 0,
            "v_bad": (ts["voltage_v"] < rules.v_min) | (ts["voltage_v"] > rules.v_max),
            "i_bad": ts["current_a"].abs() > rules.max_abs_current_a,
            "t_missing": temp.isna() | (temp == 0),
            "t_bad": (temp < rules.t_min_c) | (temp > rules.t_max_c),
            "t_spike": temp_step > rules.max_temp_step_c,
            "fingerprint": ts["voltage_v"].astype(float) * 1e3 + ts["current_a"].astype(float),
        }
    )
    agg = sample.groupby(["cell_id", "cycle_index"], sort=True).agg(
        time_not_monotonic=("neg_dt", "any"),
        voltage_out_of_range=("v_bad", "any"),
        current_out_of_range=("i_bad", "any"),
        temperature_missing=("t_missing", "mean"),
        temperature_out_of_range=("t_bad", "any"),
        temperature_spike=("t_spike", "any"),
        fingerprint=("fingerprint", "sum"),
    )
    agg = agg.reset_index().merge(cycles, on=["cell_id", "cycle_index"], how="right")

    out = pd.DataFrame({"cell_id": agg["cell_id"], "cycle_index": agg["cycle_index"]})
    empty = agg["n_samples"].fillna(0) < rules.min_samples
    out["empty_cycle"] = empty
    # identical content to the previous cycle of the same cell = duplicated export
    same_n = agg.groupby("cell_id")["n_samples"].diff() == 0
    same_fp = agg.groupby("cell_id")["fingerprint"].diff().abs() < 1e-9
    out["duplicate_cycle"] = same_n & same_fp & ~empty
    out["time_gap"] = (agg["max_time_gap_s"] > rules.max_gap_s) & ~empty
    for name in ("time_not_monotonic", "voltage_out_of_range", "current_out_of_range",
                 "temperature_out_of_range", "temperature_spike"):
        out[name] = agg[name].fillna(False).astype(bool) & ~empty
    out["temperature_missing"] = (agg["temperature_missing"].fillna(1.0) > 0.5) & ~empty
    vendor = agg["vendor_discharge_capacity_ah"]
    rel = (agg["discharge_capacity_ah"] - vendor).abs() / vendor.where(vendor > 0)
    out["capacity_mismatch"] = (rel > rules.capacity_rel_tol) & ~empty
    ce = agg["coulombic_efficiency"]
    out["ce_out_of_range"] = ((ce < rules.ce_min) | (ce > rules.ce_max)) & ~empty
    out["any_flag"] = out[list(CYCLE_FLAGS)].any(axis=1)
    out["bad_cycle"] = out[list(BAD_CYCLE_FLAGS)].any(axis=1)
    return out


def robust_z(x: pd.Series) -> pd.Series:
    """Modified z-score 0.6745·(x − median)/MAD; NaN where MAD is 0."""
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=x.index)
    return 0.6745 * (x - med) / mad


def cell_indicators(cycles: pd.DataFrame, flags: pd.DataFrame, rules: CellRules = CellRules()) -> pd.DataFrame:
    """Per-cell indicators over the early window (cycles 2-100 by default)."""
    in_win = (cycles["cycle_index"] >= rules.early_first) & (cycles["cycle_index"] <= rules.early_last)
    good = cycles.merge(flags[["cell_id", "cycle_index", "bad_cycle"]], on=["cell_id", "cycle_index"], how="left")
    win = cycles[in_win & ~good["bad_cycle"].fillna(False).to_numpy()]
    fwin = flags[(flags["cycle_index"] >= rules.early_first) & (flags["cycle_index"] <= rules.early_last)]
    rows = []
    for cell_id, g in win.groupby("cell_id"):
        g = g.sort_values("cycle_index")
        q = g["vendor_discharge_capacity_ah"].where(g["vendor_discharge_capacity_ah"] > 0, g["discharge_capacity_ah"])
        ok = q.notna()
        slope = np.polyfit(g.loc[ok, "cycle_index"], q[ok], 1)[0] if ok.sum() >= 10 else np.nan
        f = fwin[fwin["cell_id"] == cell_id]
        temp_fault = f[["temperature_missing", "temperature_out_of_range", "temperature_spike"]].any(axis=1)
        rows.append(
            {
                "cell_id": cell_id,
                "fade_slope_mah_per_cycle": slope * 1e3,
                "temperature_max_mean_c": g["temperature_max_c"].mean(),
                "charge_time_mean_min": g["charge_time_min"].mean(),
                "ir_mean_mohm": g["ir_ohm"].where(g["ir_ohm"] > 0).mean() * 1e3,
                "dcir_proxy_mean_mohm": g["dcir_proxy_ohm"].mean() * 1e3,
                "ce_median": g["coulombic_efficiency"].median(),
                "temp_fault_frac": float(temp_fault.mean()) if len(f) else np.nan,
                "flagged_cycle_frac": float(f["any_flag"].mean()) if len(f) else np.nan,
                "bad_cycle_frac": float(f["bad_cycle"].mean()) if len(f) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def cell_flags(
    indicators: pd.DataFrame, rules: CellRules = CellRules(), covariate: str = "charge_time_mean_min"
) -> pd.DataFrame:
    """Flag anomalous cells with robust z-scores and an isolation forest.

    Call once per source: cells are only compared with cells of the same
    test campaign. Temperature is compared after removing the part a linear
    fit on ``covariate`` explains (charge time for MATR, where faster
    charging runs hotter; ambient temperature for NASA), so a cell is not
    called anomalous just for its test condition.
    """
    ind = indicators.copy()
    t_ok = ind[["temperature_max_mean_c", covariate]].notna().all(axis=1)
    ind["temperature_residual_c"] = np.nan
    if t_ok.sum() >= 5:
        b, a = np.polyfit(ind.loc[t_ok, covariate], ind.loc[t_ok, "temperature_max_mean_c"], 1)
        ind.loc[t_ok, "temperature_residual_c"] = ind.loc[t_ok, "temperature_max_mean_c"] - (
            a + b * ind.loc[t_ok, covariate]
        )

    zcols = {
        "fade_slope_mah_per_cycle": "z_fade",
        "temperature_residual_c": "z_temperature",
        "ir_mean_mohm": "z_ir",
        "ce_median": "z_ce",
    }
    for col, z in zcols.items():
        ind[z] = robust_z(ind[col])
    ind["flag_fade"] = ind["z_fade"].abs() > rules.robust_z
    ind["flag_temperature"] = ind["z_temperature"].abs() > rules.robust_z
    ind["flag_ir"] = ind["z_ir"].abs() > rules.robust_z
    ind["flag_ce"] = ind["z_ce"].abs() > rules.robust_z
    ind["flag_temperature_sensor"] = ind["temp_fault_frac"] > rules.temp_fault_frac

    features = ["fade_slope_mah_per_cycle", "temperature_residual_c", "ir_mean_mohm", "ce_median",
                "temp_fault_frac", "bad_cycle_frac"]
    x = ind[features].astype(float)
    x = x.fillna(x.median())
    ind["iforest_score"] = np.nan
    ind["flag_iforest"] = False
    if len(x) >= 10:
        scaled = RobustScaler().fit_transform(x)
        model = IsolationForest(
            n_estimators=500, contamination=rules.iforest_contamination, random_state=rules.iforest_seed
        ).fit(scaled)
        ind["iforest_score"] = -model.score_samples(scaled)  # higher = more anomalous
        ind["flag_iforest"] = model.predict(scaled) == -1
    flag_cols = [c for c in ind.columns if c.startswith("flag_")]
    ind["n_flags"] = ind[flag_cols].sum(axis=1)
    ind["anomalous"] = ind[flag_cols].any(axis=1)
    return ind


def rules_dict() -> dict:
    return {"cycle": {k: asdict(v) for k, v in RULES_BY_SOURCE.items()}, "cell": asdict(CellRules())}
