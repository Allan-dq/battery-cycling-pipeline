"""Pipeline throughput metrics in Prometheus format.

Each stage records rows/cycles/cells processed and its wall time. The CLI
writes them to ``<root>/pipeline_metrics.prom`` (node-exporter textfile
format), which the API also serves at ``/metrics``. Runs that execute only
some stages keep the other stages' last values (state in
``pipeline_metrics.json``), so ``bcp report`` does not wipe the numbers of the
last full run.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

from prometheus_client import CollectorRegistry, Counter, Gauge, write_to_textfile

REGISTRY = CollectorRegistry()
ROWS = Counter("bcp_rows_processed", "Time-series rows processed", ["stage"], registry=REGISTRY)
CYCLES = Counter("bcp_cycles_processed", "Cycles processed", ["stage"], registry=REGISTRY)
CELLS = Counter("bcp_cells_processed", "Cells processed", ["stage"], registry=REGISTRY)
DURATION = Gauge(
    "bcp_stage_duration_seconds", "Wall time of the last run of a stage", ["stage"], registry=REGISTRY
)
ROWS_PER_S = Gauge(
    "bcp_stage_rows_per_second", "Rows per second in the last run", ["stage"], registry=REGISTRY
)
LAST_SUCCESS = Gauge(
    "bcp_stage_last_success_unixtime", "When the stage last finished", ["stage"], registry=REGISTRY
)


@contextmanager
def stage(name: str):
    """Time a stage; the body may call ``ROWS.labels(name).inc(n)`` etc."""
    start = time.perf_counter()
    rows_before = ROWS.labels(name)._value.get()
    yield
    elapsed = time.perf_counter() - start
    DURATION.labels(name).set(elapsed)
    rows = ROWS.labels(name)._value.get() - rows_before
    if elapsed > 0 and rows:
        ROWS_PER_S.labels(name).set(rows / elapsed)
    LAST_SUCCESS.labels(name).set_to_current_time()


def _this_run() -> dict[str, dict[str, float]]:
    """Per-stage values recorded in this process."""
    out: dict[str, dict[str, float]] = {}
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name.endswith("_created"):
                continue
            stage_name = sample.labels.get("stage")
            if stage_name:
                out.setdefault(stage_name, {})[sample.name] = sample.value
    return out


def write(path: Path) -> Path:
    """Merge this run's stages into the saved state and render the textfile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state_path = path.with_suffix(".json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state.update(_this_run())
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    reg = CollectorRegistry()
    gauges: dict[str, Gauge] = {}
    for stage_name, values in sorted(state.items()):
        for name, value in sorted(values.items()):
            if name not in gauges:
                gauges[name] = Gauge(name, f"{name} (last run of each stage)", ["stage"], registry=reg)
            gauges[name].labels(stage_name).set(value)
    write_to_textfile(str(path), reg)
    return path
