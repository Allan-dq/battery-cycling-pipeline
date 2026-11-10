from battery_pipeline import telemetry


def test_partial_runs_keep_other_stages(tmp_path):
    path = tmp_path / "pipeline_metrics.prom"
    with telemetry.stage("unit_a"):
        telemetry.ROWS.labels("unit_a").inc(100)
    telemetry.write(path)
    # a later run in a fresh process would only have unit_b; simulate by dropping unit_a from the state
    with telemetry.stage("unit_b"):
        telemetry.ROWS.labels("unit_b").inc(5)
    telemetry.write(path)
    text = path.read_text()
    assert 'bcp_rows_processed_total{stage="unit_a"} 100.0' in text
    assert 'bcp_rows_processed_total{stage="unit_b"} 5.0' in text
    assert "bcp_stage_duration_seconds" in text
