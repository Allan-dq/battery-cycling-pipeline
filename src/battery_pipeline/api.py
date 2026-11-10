"""Read-only REST API over the warehouse.

    uvicorn battery_pipeline.api:app            # BCP_ROOT=data/processed by default

GET /health                      liveness + which tables are loaded
GET /cells                       one row per cell (cell_report view)
GET /cells/{cell_id}             cell metadata, QC result, features, predictions
GET /cells/{cell_id}/cycles      per-cycle metrics (?start=&end= to slice)
GET /metrics                     Prometheus: API request counts + the last pipeline run's
                                 throughput (the textfile `bcp` writes to <root>/pipeline_metrics.prom)
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, generate_latest

from battery_pipeline.store import Lake

API_REGISTRY = CollectorRegistry()
REQUESTS = Counter("bcp_api_requests", "API requests", ["endpoint"], registry=API_REGISTRY)

app = FastAPI(title="battery-cycling-pipeline", version="0.1.0")


@lru_cache(maxsize=1)
def lake() -> Lake:
    return Lake(Path(os.environ.get("BCP_ROOT", "data/processed")))


def _query(sql: str, params: list | None = None) -> list[dict]:
    con = lake().connect(read_only=True)
    try:
        cur = con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [{c: _clean(v) for c, v in zip(cols, row, strict=True)} for row in cur.fetchall()]
    finally:
        con.close()


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _tables() -> set[str]:
    return {r["table_name"] for r in _query("SELECT table_name FROM information_schema.tables")}


@app.get("/health")
def health() -> dict:
    REQUESTS.labels("health").inc()
    if not lake().warehouse.exists():
        raise HTTPException(503, "warehouse not built yet; run `bcp all`")
    return {"status": "ok", "tables": sorted(_tables())}


@app.get("/cells")
def cells() -> list[dict]:
    REQUESTS.labels("cells").inc()
    return _query("SELECT * FROM cell_report ORDER BY cell_id")


@app.get("/cells/{cell_id}")
def cell(cell_id: str) -> dict:
    REQUESTS.labels("cell").inc()
    rows = _query("SELECT * FROM cell_report WHERE cell_id = ?", [cell_id])
    if not rows:
        raise HTTPException(404, f"unknown cell {cell_id!r}")
    out = {"cell": rows[0]}
    tables = _tables()
    if "features" in tables:
        f = _query("SELECT * FROM features WHERE cell_id = ?", [cell_id])
        out["features"] = f[0] if f else None
    if "predictions" in tables:
        out["predictions"] = _query(
            "SELECT model, split, cycle_life, predicted FROM predictions WHERE cell_id = ? ORDER BY model",
            [cell_id],
        )
    return out


@app.get("/cells/{cell_id}/cycles")
def cycles(cell_id: str, start: int = Query(1, ge=1), end: int | None = Query(None, ge=1)) -> list[dict]:
    REQUESTS.labels("cycles").inc()
    rows = _query(
        "SELECT * FROM cycles WHERE cell_id = ? AND cycle_index >= ? AND cycle_index <= ? ORDER BY cycle_index",
        [cell_id, start, end or 10**9],
    )
    if not rows and not _query("SELECT 1 FROM cells WHERE cell_id = ?", [cell_id]):
        raise HTTPException(404, f"unknown cell {cell_id!r}")
    return rows


@app.get("/metrics")
def metrics() -> Response:
    REQUESTS.labels("metrics").inc()
    body = generate_latest(API_REGISTRY).decode()
    prom = lake().root / "pipeline_metrics.prom"
    if prom.exists():
        body += prom.read_text()
    return Response(body, media_type=CONTENT_TYPE_LATEST)
