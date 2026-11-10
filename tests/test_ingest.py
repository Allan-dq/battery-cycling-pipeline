"""Loaders against the committed (trimmed, real) fixtures."""

import numpy as np
import pytest

from battery_pipeline.ingest import ingest_matr, matr, nasa
from battery_pipeline.metrics import cycle_metrics
from battery_pipeline.store import Lake
from tests.conftest import FIXTURES

pytestmark = pytest.mark.integration


def test_matr_reader_scales_current_to_amperes():
    cell = next(matr.read_cells(FIXTURES / "matr_sample_b1.h5", {"b1c1"}))
    assert cell.cell_id == "b1c1"
    assert cell.charge_policy  # e.g. "3.6C(80%)-3.6C"
    m = cycle_metrics(cell.timeseries, cell.summary.assign(cell_id="b1c1")).set_index("cycle_index")
    # coulomb counting on the thinned raw current agrees with the dataset's own capacity
    rel = (m.loc[2, "discharge_capacity_ah"] - m.loc[2, "vendor_discharge_capacity_ah"]) / m.loc[
        2, "vendor_discharge_capacity_ah"
    ]
    assert abs(rel) < 0.02
    assert cell.timeseries["current_a"].abs().max() > 4.0  # 4C discharge = 4.4 A, not 4.0 "C"


def test_cycle_one_is_the_empty_placeholder():
    cell = next(matr.read_cells(FIXTURES / "matr_sample_b1.h5", {"b1c1"}))
    n1 = (cell.timeseries["cycle_index"] == 1).sum()
    assert n1 <= 2


def test_continued_cell_is_merged(tmp_path):
    lake = Lake(tmp_path)
    files = [FIXTURES / "matr_sample_b2.h5", FIXTURES / "matr_sample_b1.h5"]  # order must not matter
    report = ingest_matr(files, lake, log=lambda m: None)
    assert "b2c7->b1c0" in report.merged_continuations
    cells = lake.read_table("cells").set_index("cell_id")
    assert "b2c7" not in cells.index
    assert cells.loc["b1c0", "n_cycles"] == 60 + 50
    b1 = next(matr.read_cells(FIXTURES / "matr_sample_b1.h5", {"b1c0"}))
    assert cells.loc["b1c0", "reported_cycle_life"] == b1.reported_cycle_life + 662
    ts = lake.read_timeseries("b1c0", cycles=[60, 61, 95])
    n = ts.groupby("cycle_index").size()
    # merged numbering: b2c7's cycle 35 (data in the fixture) becomes b1c0's cycle 95;
    # 60/61 are two-sample placeholders in the trimmed fixture
    assert n[95] > 100 and n[60] <= 2 and n[61] <= 2
    assert (
        ts.loc[ts.cycle_index == 95, "test_time_s"].min() >= ts.loc[ts.cycle_index == 60, "test_time_s"].max()
    )
    vendor = lake.read_table("vendor_summary")
    assert vendor.loc[vendor.cell_id == "b1c0", "cycle_index"].max() == 110


def test_nasa_reader_builds_charge_plus_discharge_cycles():
    files = nasa.find_files(FIXTURES / "nasa")
    assert [f.stem for f in files] == ["B0005", "B0031", "B0047"]
    cell = nasa.read_cell(files[0])
    ts = cell.timeseries
    assert cell.cell_id == "nasa_B0005" and cell.ambient_c == "24"
    first = ts[ts.cycle_index == 2]
    assert (first["step_type"] == "charge").any() and (first["step_type"] == "discharge").any()
    assert np.all(np.diff(first["cycle_time_s"]) >= 0)
    m = cycle_metrics(ts, cell.summary.assign(cell_id=cell.cell_id)).set_index("cycle_index")
    rel = (m["discharge_capacity_ah"] - m["vendor_discharge_capacity_ah"]).abs() / m[
        "vendor_discharge_capacity_ah"
    ]
    assert rel.median() < 0.02


def test_nasa_missing_capacity_stays_missing():
    # B0050/B0052 have discharge records with an empty Capacity field
    assert np.isnan(nasa._first(np.array([])))
    assert np.isnan(nasa._first(None))
    assert nasa._first(np.array([[1.85]])) == 1.85


def test_cycle_life_rule_matches_the_authors_loader():
    q = np.r_[0.0, np.linspace(1.08, 0.8803, 99)]  # stopped just above 0.88 after 100 cycles
    assert matr.cycle_life_from_summary(q) == 101
    q2 = np.r_[0.0, 1.07, 1.0, 0.879, 0.87]  # crosses 0.88 at cycle 4
    assert matr.cycle_life_from_summary(q2) == 4
    q3 = np.r_[0.0, 1.07, 1.0, 0.95]  # stopped early, never reached 80%
    assert np.isnan(matr.cycle_life_from_summary(q3))
