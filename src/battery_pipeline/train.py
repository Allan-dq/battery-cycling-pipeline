"""Stage 3: assemble the modelling table, fit, track in MLflow."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from battery_pipeline import model, telemetry
from battery_pipeline.cohort import paper_split
from battery_pipeline.store import Lake


def modelling_table(lake: Lake, features_table: str = "features") -> tuple[pd.DataFrame, str | None]:
    """Features + split + target (dataset-reported cycle life, as in the paper)."""
    cells = lake.read_table("cells")
    feats = lake.read_table(features_table)
    split = paper_split(cells)
    data = split.merge(feats, on="cell_id", how="left").merge(
        cells[["cell_id", "reported_cycle_life"]], on="cell_id", how="left"
    )
    data = data.rename(columns={"reported_cycle_life": "cycle_life"})
    test1 = data[data["split"] == "test1"].dropna(subset=["cycle_life"])
    # the paper's parenthetical numbers drop the primary-test cell that reaches 80% fastest
    outlier = str(test1.loc[test1["cycle_life"].idxmin(), "cell_id"]) if len(test1) else None
    return data, outlier


def run(lake: Lake, tracking_dir: Path, log: Callable[[str], None] = print, register: bool = True) -> dict:
    with telemetry.stage("train"):
        data, outlier = modelling_table(lake)
        data_noqc, _ = modelling_table(lake, "features_noqc")
        log(f"  cohort: {data['split'].value_counts().to_dict()}, primary-test outlier = {outlier}")
        results = model.run(data, outlier, tracking_dir, log=log, register=register, data_noqc=data_noqc)
        lake.write_table("predictions", results.pop("predictions"))
        lake.write_table("modelling_table", data)
        (lake.root / "model_results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
