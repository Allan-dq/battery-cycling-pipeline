"""Whole CLI on the committed fixtures: ingest -> process -> warehouse -> train -> report -> API."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def test_all_stages_write_their_outputs(sample_run):
    for name in (
        "cells",
        "cycles",
        "qc_cycle_flags",
        "qc_cell_flags",
        "features",
        "features_noqc",
        "predictions",
    ):
        assert (sample_run / f"{name}.parquet").exists(), name
    assert (sample_run / "warehouse.duckdb").exists()
    assert (sample_run / "pipeline_metrics.prom").read_text().count("bcp_rows_processed") >= 2
    results = json.loads((sample_run / "reports" / "results.json").read_text())
    assert results["dataset"]["cells"] == 13 + 3  # 13 MATR (b2c7 merged into b1c0) + 3 NASA
    assert "variance" in results["models"]["models"]
    assert results["models"]["selected_candidate"] in {
        "gbr_candidates",
        "enet_candidates_temp",
        "enet_discharge_ica",
    }
    assert (sample_run / "reports" / "results.md").read_text().startswith("<!-- generated")


def test_warehouse_views_answer_sql(sample_run):
    from battery_pipeline.store import Lake

    con = Lake(sample_run).connect()
    try:
        n_ts = con.execute("SELECT count(DISTINCT cell_id) FROM timeseries").fetchone()[0]
        report = con.execute("SELECT cell_id, anomalous FROM cell_report").fetchall()
        errors = con.execute("SELECT model, split, rmse_cycles FROM prediction_errors").fetchall()
        empties = con.execute("SELECT sum(empty_cycles) FROM cycle_quality").fetchone()[0]
    finally:
        con.close()
    assert n_ts == 16
    assert len(report) == 16
    assert errors
    assert empties > 0  # fixture cycles 13-94 are placeholders and must be caught


def test_mlflow_has_runs_and_registered_models(sample_run):
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(f"sqlite:///{(sample_run / 'mlflow.db').as_posix()}")
    client = MlflowClient()
    exp = client.get_experiment_by_name("cycle-life-prediction")
    runs = client.search_runs([exp.experiment_id])
    assert {r.info.run_name for r in runs} >= {"variance", "discharge", "full", "gbr_candidates"}
    assert all("rmse_train" in r.data.metrics for r in runs)
    names = {m.name for m in client.search_registered_models()}
    assert {"battery-cycle-life-baseline", "battery-cycle-life-candidate"} <= names
    import skops.io as sio

    local = mlflow.artifacts.download_artifacts("models:/battery-cycle-life-baseline/1")
    model_file = next(Path(local).rglob("*.skops"))
    model = sio.load(model_file, trusted=sio.get_untrusted_types(file=model_file))
    assert hasattr(model, "predict")


def test_api_serves_cells_cycles_and_metrics(sample_run, monkeypatch):
    from fastapi.testclient import TestClient

    from battery_pipeline import api

    monkeypatch.setenv("BCP_ROOT", str(sample_run))
    api.lake.cache_clear()
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    cells = client.get("/cells").json()
    assert {c["cell_id"] for c in cells} >= {"b1c0", "nasa_B0005"}
    one = client.get("/cells/b1c1").json()
    assert one["cell"]["cell_id"] == "b1c1" and one["features"]["dq_var"] is not None
    cyc = client.get("/cells/b1c1/cycles", params={"start": 2, "end": 5}).json()
    assert [c["cycle_index"] for c in cyc] == [2, 3, 4, 5]
    assert client.get("/cells/nope").status_code == 404
    metrics = client.get("/metrics").text
    assert "bcp_api_requests_total" in metrics and "bcp_rows_processed_total" in metrics
