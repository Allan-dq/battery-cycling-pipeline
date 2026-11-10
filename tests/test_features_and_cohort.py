import numpy as np
import pandas as pd
import pytest

from battery_pipeline import features as F
from battery_pipeline.cohort import PAPER_REMOVED, paper_split
from battery_pipeline.schema import TIMESERIES, SchemaError, conform


def _summary(n=110, fade=-1e-4):
    c = np.arange(1, n + 1)
    return pd.DataFrame(
        {
            "cycle_index": c,
            "vendor_discharge_capacity_ah": np.where(c == 1, 0.0, 1.08 + fade * c),
            "vendor_charge_time_min": 10.0,
            "vendor_temperature_max_c": 35.0,
            "vendor_temperature_min_c": 30.0,
            "ir_ohm": 0.016 + 1e-6 * c,
        }
    )


def _metrics(n=110):
    return pd.DataFrame({"cycle_index": np.arange(1, n + 1), "temperature_integral_c_min": 1700.0})


def test_delta_q_features_follow_their_definitions():
    rng = np.random.default_rng(1)
    q = np.tile(np.linspace(0, 1.1, 1000), (110, 1))
    dq = rng.normal(-0.01, 0.003, 1000)
    q[99] = q[9] + dq  # cycle 100 = cycle 10 + dq
    f = F.cell_features(q, None, _summary(), _metrics())
    assert f["dq_var"] == pytest.approx(np.log10(np.var(dq, ddof=1)))
    assert f["dq_min"] == pytest.approx(np.log10(abs(dq.min())))
    assert f["dq_2v"] == pytest.approx(np.log10(abs(dq[-1])))
    assert f["fade_slope_2_100"] == pytest.approx(-1e-4)
    assert f["qd2"] == pytest.approx(1.08 - 2e-4)
    assert f["charge_time_first5"] == 10.0
    assert f["t_integral_2_100"] == pytest.approx(1700.0 * 99)
    assert f["ir_diff_100_2"] == pytest.approx(98e-6)


def test_bad_cycles_are_dropped_and_dq_moves_to_a_neighbour():
    q = np.tile(np.linspace(0, 1.1, 50), (110, 1))
    s = _summary()
    s.loc[s.cycle_index == 50, "vendor_discharge_capacity_ah"] = 1.6  # broken summary value
    good = set(range(2, 111)) - {50, 100}
    clean = F.cell_features(q, None, s, _metrics(), good_cycles=good)
    raw = F.cell_features(q, None, s, _metrics(), good_cycles=None)
    assert raw["qd_max_minus_2"] > 0.5
    assert clean["qd_max_minus_2"] < 0.01
    assert clean["dq_cycles"] in {"99-10", "101-10"}
    assert clean["n_bad_cycles_2_100"] == 2


def test_feature_lists_match_the_paper_counts():
    assert len(F.CANDIDATE_FEATURES) == 20
    assert len(F.DISCHARGE_FEATURES) == 6
    assert len(F.FULL_FEATURES) == 9
    assert set(F.FULL_FEATURES) <= set(F.CANDIDATE_FEATURES)


def test_paper_split_sizes_and_order():
    ids = [f"b1c{i}" for i in range(46)] + [f"b2c{i}" for i in range(48) if i not in (7, 8, 9, 15, 16)]
    ids += [f"b3c{i}" for i in range(46)]
    split = paper_split(pd.DataFrame({"cell_id": ids, "source": "matr"}))
    assert split["split"].value_counts().to_dict() == {"test1": 43, "train": 41, "test2": 40}
    assert not set(split["cell_id"]) & set(PAPER_REMOVED)
    s = split.set_index("cell_id")["split"]
    assert s["b1c0"] == "test1" and s["b1c1"] == "train"
    assert s["b2c47"] == "test1"  # position 83 joins the test set
    assert s["b2c1"] == "test1"


def test_paper_split_ignores_other_sources():
    cells = pd.DataFrame({"cell_id": ["b1c0", "b1c1", "nasa_B0005"], "source": ["matr", "matr", "nasa"]})
    assert "nasa_B0005" not in set(paper_split(cells)["cell_id"])


def test_conform_rejects_missing_and_extra_columns():
    df = pd.DataFrame({c: [] for c in TIMESERIES.names})
    assert conform(df, TIMESERIES).schema == TIMESERIES
    with pytest.raises(SchemaError):
        conform(df.drop(columns=["voltage_v"]), TIMESERIES)
    with pytest.raises(SchemaError):
        conform(df.assign(extra=[]), TIMESERIES)
