import numpy as np
import pandas as pd
import pytest

from battery_pipeline.metrics import cycle_metrics, delta_q, discharge_curve, dqdv
from battery_pipeline.schema import CYCLES
from battery_pipeline.steps import label_steps
from tests.conftest import cc_cycle


def test_label_steps_segments_charge_rest_discharge():
    types, index = label_steps(np.array([0.0, 1.0, 1.0, 0.001, 0.0, -2.0, -2.0, 0.0]))
    assert list(types) == ["rest", "charge", "charge", "rest", "rest", "discharge", "discharge", "rest"]
    assert list(index) == [1, 2, 2, 3, 3, 4, 4, 5]


def test_coulomb_counting_matches_known_capacity():
    ts = cc_cycle(charge_a=1.0, discharge_a=2.0, charge_s=3600)  # 1 Ah in, 1 Ah out
    m = cycle_metrics(ts)
    assert list(m.columns) == CYCLES.names
    row = m.iloc[0]
    assert row.charge_capacity_ah == pytest.approx(1.0, rel=2e-3)
    assert row.discharge_capacity_ah == pytest.approx(1.0, rel=2e-3)
    assert row.coulombic_efficiency == pytest.approx(1.0, rel=5e-3)
    # 1 Ah at 3.4 V in, 1 Ah at 3.1 V out
    assert row.energy_efficiency == pytest.approx(3.1 / 3.4, rel=5e-3)
    assert row.charge_time_min == pytest.approx(60.0, abs=0.2)
    assert row.temperature_mean_c == pytest.approx(30.0)


def test_dcir_proxy_is_voltage_step_over_current_step():
    ts = cc_cycle()  # rest at 3.3 V -> discharge at 3.1 V and -2 A
    m = cycle_metrics(ts)
    assert m.iloc[0].dcir_proxy_ohm == pytest.approx(0.2 / 2.0)


def test_vendor_columns_are_joined():
    ts = cc_cycle(cell_id="x", cycle=3)
    vendor = pd.DataFrame(
        {"cell_id": ["x"], "cycle_index": [3], "vendor_discharge_capacity_ah": [0.99], "ir_ohm": [0.016]}
    )
    m = cycle_metrics(ts, vendor)
    assert m.iloc[0].vendor_discharge_capacity_ah == 0.99
    assert m.iloc[0].ir_ohm == 0.016
    assert np.isnan(m.iloc[0].vendor_charge_capacity_ah)


def test_delta_q_uses_one_based_cycles():
    q = np.array([[0, 0, 0], [1.0, 1.0, 1.0], [1.1, 1.2, 1.3]])
    np.testing.assert_allclose(delta_q(q, 3, 2), [0.1, 0.2, 0.3])


def test_dqdv_peak_sits_on_the_plateau():
    # capacity rises steeply around 3.2 V (a voltage plateau), flat elsewhere
    v = np.linspace(3.5, 2.5, 600)
    q = 1.0 / (1.0 + np.exp((v - 3.2) / 0.01))
    curve = dqdv(v, q, resolution_v=0.005)
    peak_v = curve.loc[curve["dqdv"].idxmax(), "voltage"]
    assert peak_v == pytest.approx(3.2, abs=0.02)
    assert (curve["dqdv"] >= 0).all()


def test_discharge_curve_selects_discharge_samples():
    ts = cc_cycle()
    v, q = discharge_curve(ts)
    assert len(v) == (ts["step_type"] == "discharge").sum()
    assert np.all(np.diff(q) >= 0)
