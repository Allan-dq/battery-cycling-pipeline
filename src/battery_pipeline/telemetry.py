"""Pipeline throughput metrics in Prometheus format.

Each stage records rows/cycles/cells processed and its wall time. The CLI
writes them to ``<root>/pipeline_metrics.prom`` (node-exporter textfile
format) and the API serves the same registry at ``/metrics``.
"""

from __future__ import annotations

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


def write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_to_textfile(str(path), REGISTRY)
    return path
