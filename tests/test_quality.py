import numpy as np
import pandas as pd

from battery_pipeline import quality
from battery_pipeline.metrics import cycle_metrics
from tests.conftest import cc_cycle


def _flags(ts):
    cyc = cycle_metrics(ts)
    return quality.cycle_flags(ts, cyc).set_index("cycle_index")


def test_clean_cycle_has_no_flags():
    f = _flags(cc_cycle())
    assert not f.loc[1, list(quality.CYCLE_FLAGS)].any()


def test_injected_problems_are_flagged():
    clean = cc_cycle(cycle=1)
    spike = cc_cycle(cycle=2)
    spike.loc[100, "temperature_c"] = 45.0  # one-sample thermocouple spike
    gap = cc_cycle(cycle=3)
    gap.loc[780:, ["cycle_time_s", "test_time_s"]] += 900.0  # 15 min logging gap during the rest step
    volts = cc_cycle(cycle=4)
    volts.loc[50, "voltage_v"] = 0.0
    dup = cc_cycle(cycle=5)  # identical content to cycle 4 except the glitch below
    dup.loc[50, "voltage_v"] = 0.0
    empty = cc_cycle(cycle=6).head(2)
    no_temp = cc_cycle(cycle=7)
    no_temp["temperature_c"] = np.nan
    f = _flags(pd.concat([clean, spike, gap, volts, dup, empty, no_temp], ignore_index=True))
    assert f.loc[2, "temperature_spike"] and not f.loc[1, "temperature_spike"]
    assert f.loc[3, "time_gap"]
    assert f.loc[4, "voltage_out_of_range"]
    assert f.loc[5, "duplicate_cycle"] and not f.loc[4, "duplicate_cycle"]
    assert f.loc[6, "empty_cycle"] and f.loc[6, "bad_cycle"]
    assert f.loc[7, "temperature_missing"]
    assert not f.loc[3, "bad_cycle"]  # a pause alone does not make the cycle unusable


def test_capacity_mismatch_is_relative_to_the_cells_own_offset():
    ts = pd.concat([cc_cycle(cell_id="x", cycle=k) for k in range(1, 8)], ignore_index=True)
    # the dataset reads 1.5% low on every cycle (calibration), and 10% low on cycle 5 (broken)
    vendor = pd.DataFrame({"cell_id": "x", "cycle_index": range(1, 8), "vendor_discharge_capacity_ah": 0.985})
    vendor.loc[vendor.cycle_index == 5, "vendor_discharge_capacity_ah"] = 0.90
    f = quality.cycle_flags(ts, cycle_metrics(ts, vendor)).set_index("cycle_index")
    assert f.loc[5, "capacity_mismatch"] and f.loc[5, "bad_cycle"]
    assert not f.drop(index=5)["capacity_mismatch"].any()
    assert f.loc[1, "capacity_rel_diff"] > 0.01


def test_robust_z_ignores_the_outlier_it_scores():
    x = pd.Series([1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 10.0])
    z = quality.robust_z(x)
    assert z.iloc[-1] > 3.5
    assert (z.iloc[:-1].abs() < 3.5).all()


def test_cell_flags_catch_a_fast_fading_hot_cell():
    rng = np.random.default_rng(0)
    n = 30
    ind = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(n)],
            "fade_slope_mah_per_cycle": rng.normal(-0.02, 0.005, n),
            "temperature_max_mean_c": rng.normal(36, 0.3, n),
            "charge_time_mean_min": rng.normal(30, 2, n),
            "ir_mean_mohm": rng.normal(16.5, 0.3, n),
            "dcir_proxy_mean_mohm": rng.normal(19.5, 0.3, n),
            "ce_median": rng.normal(0.999, 0.0003, n),
            "temp_fault_frac": np.zeros(n),
            "flagged_cycle_frac": np.zeros(n),
            "bad_cycle_frac": np.zeros(n),
        }
    )
    ind.loc[7, "fade_slope_mah_per_cycle"] = -0.3
    ind.loc[7, "temperature_max_mean_c"] = 41.0
    ind.loc[12, "temp_fault_frac"] = 0.4
    out = quality.cell_flags(ind).set_index("cell_id")
    assert out.loc["c7", "flag_fade"] and out.loc["c7", "flag_temperature"]
    assert out.loc["c12", "flag_temperature_sensor"]
    assert out.loc["c7", "flag_iforest"]
    assert out["anomalous"].sum() <= 5


def test_temperature_is_compared_after_the_charge_time_effect():
    # hotter cells charge faster; a hot fast-charged cell is normal, not anomalous
    n = 20
    ct = np.linspace(20, 40, n)
    ind = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(n)],
            "fade_slope_mah_per_cycle": -0.02 + 0.001 * np.sin(np.arange(n)),
            "temperature_max_mean_c": 45 - 0.3 * ct + 0.05 * np.cos(np.arange(n)),
            "charge_time_mean_min": ct,
            "ir_mean_mohm": 16.5 + 0.1 * np.sin(np.arange(n) * 2),
            "ce_median": 0.999 + 1e-4 * np.cos(np.arange(n) * 3),
            "temp_fault_frac": np.zeros(n),
            "bad_cycle_frac": np.zeros(n),
        }
    )
    out = quality.cell_flags(ind)
    assert not out["flag_temperature"].any()


def test_rules_differ_by_source():
    assert quality.rules_for("nasa_B0005").v_max > 4.2
    assert quality.rules_for("b1c0").v_max < 4.0
