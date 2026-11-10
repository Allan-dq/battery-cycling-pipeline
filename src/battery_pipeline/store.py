"""Parquet lake + DuckDB warehouse.

Layout under one root (``data/processed`` by default)::

    timeseries/cell_id=<id>/part-<batch>.parquet   canonical TIMESERIES rows
    curves/cell_id=<id>/part-<batch>.parquet       Q(V), T(V) on the voltage grid
    vendor_summary.parquet                         per-cycle values the dataset ships
    cells.parquet, cycles.parquet, qc_*.parquet, features.parquet, predictions.parquet
    warehouse.duckdb                               tables + views over the above

The time series stays in Parquet (hive-partitioned by cell) and DuckDB reads
it in place, so the warehouse file holds only the small derived tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DERIVED_TABLES = (
    "cells",
    "vendor_summary",
    "cycles",
    "qc_cycle_flags",
    "qc_cell_flags",
    "features",
    "predictions",
)


@dataclass(frozen=True)
class Lake:
    root: Path

    @property
    def timeseries(self) -> Path:
        return self.root / "timeseries"

    @property
    def curves(self) -> Path:
        return self.root / "curves"

    @property
    def warehouse(self) -> Path:
        return self.root / "warehouse.duckdb"

    def table(self, name: str) -> Path:
        return self.root / f"{name}.parquet"

    def partition(self, kind: str, cell_id: str, part: str) -> Path:
        folder = getattr(self, kind) / f"cell_id={cell_id}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"part-{part}.parquet"

    # -- io ------------------------------------------------------------------
    def write_partition(self, kind: str, cell_id: str, part: str, table: pa.Table) -> Path:
        path = self.partition(kind, cell_id, part)
        # cell_id lives in the directory name (hive partitioning), not in the file
        if "cell_id" in table.column_names:
            table = table.drop(["cell_id"])
        pq.write_table(table, path, compression="zstd")
        return path

    def write_table(self, name: str, df: pd.DataFrame) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.table(name)
        df.to_parquet(path, index=False, compression="zstd")
        return path

    def read_table(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.table(name))

    def read_timeseries(self, cell_id: str, cycles: list[int] | None = None) -> pd.DataFrame:
        files = sorted((self.timeseries / f"cell_id={cell_id}").glob("*.parquet"))
        filters = [("cycle_index", "in", list(cycles))] if cycles else None
        frames = [pq.read_table(p, filters=filters).to_pandas() for p in files]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(df):
            df.insert(0, "cell_id", cell_id)
        return df

    def read_curves(self, cell_id: str) -> pd.DataFrame:
        files = sorted((self.curves / f"cell_id={cell_id}").glob("*.parquet"))
        if not files:  # sources without Q(V) curves (NASA)
            return pd.DataFrame(columns=["cycle_index", "qdlin", "tdlin"])
        df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        return df.sort_values("cycle_index").reset_index(drop=True)

    def cell_ids(self) -> list[str]:
        return sorted(p.name.split("=", 1)[1] for p in self.timeseries.glob("cell_id=*"))

    # -- warehouse -------------------------------------------------------------
    def build_warehouse(self) -> Path:
        """(Re)create warehouse.duckdb: derived tables + views from sql/views.sql."""
        if self.warehouse.exists():
            self.warehouse.unlink()
        con = duckdb.connect(str(self.warehouse))
        try:
            for name in DERIVED_TABLES:
                path = self.table(name)
                if path.exists():
                    con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_parquet('{path.as_posix()}')")
            ts_glob = (self.timeseries / "*" / "*.parquet").as_posix()
            con.execute(
                "CREATE VIEW timeseries AS SELECT * FROM "
                f"read_parquet('{ts_glob}', hive_partitioning = true, union_by_name = true)"
            )
            views = resources.files("battery_pipeline").joinpath("sql/views.sql").read_text()
            for statement in _split_sql(views):
                present = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
                if _referenced_tables(statement) <= present:
                    con.execute(statement)
        finally:
            con.close()
        return self.warehouse

    def connect(self, read_only: bool = True) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.warehouse), read_only=read_only)


def _split_sql(text: str) -> list[str]:
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def _referenced_tables(statement: str) -> set[str]:
    words = {w.strip("(),").lower() for w in statement.replace("\n", " ").split()}
    return {t for t in (*DERIVED_TABLES, "timeseries") if t in words}
