from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

FIXTURES = Path(__file__).parent / "fixtures"


def cc_cycle(
    cell_id: str = "c",
    cycle: int = 1,
    charge_a: float = 1.0,
    discharge_a: float = 2.0,
    charge_s: float = 3600.0,
    rest_s: float = 600.0,
    dt: float = 5.0,
    temperature: float = 30.0,
) -> pd.DataFrame:
    """A synthetic constant-current cycle: charge, rest, discharge (known answers)."""
    discharge_s = charge_a * charge_s / discharge_a
    t = np.arange(0.0, charge_s + rest_s + discharge_s + dt / 2, dt)
    current = np.where(t < charge_s, charge_a, np.where(t < charge_s + rest_s, 0.0, -discharge_a))
    current[-1] = 0.0  # end at rest so the last interval is not double counted
    voltage = np.where(current > 0, 3.4, np.where(current < 0, 3.1, 3.3))
    from battery_pipeline.steps import label_steps

    step_type, step_index = label_steps(current)
    q_ch = np.concatenate([[0], np.cumsum(np.clip(current[:-1], 0, None) * dt)]) / 3600
    q_dis = np.concatenate([[0], np.cumsum(np.clip(-current[:-1], 0, None) * dt)]) / 3600
    return pd.DataFrame(
        {
            "cell_id": cell_id,
            "cycle_index": np.int32(cycle),
            "step_index": step_index,
            "step_type": step_type,
            "test_time_s": t,
            "cycle_time_s": t,
            "current_a": current,
            "voltage_v": voltage,
            "temperature_c": np.full(len(t), temperature),
            "charge_capacity_ah": q_ch,
            "discharge_capacity_ah": q_dis,
        }
    )


@pytest.fixture(scope="session")
def sample_run(tmp_path_factory) -> Path:
    """Run the whole CLI once on the committed fixtures; tests share the output."""
    from battery_pipeline.cli import main

    root = tmp_path_factory.mktemp("sample")
    raw = root / "raw"
    raw.mkdir()
    for p in FIXTURES.glob("matr_sample_*.h5"):
        shutil.copy(p, raw / p.name)
    rc = main(
        [
            "all",
            "--raw",
            str(raw),
            "--nasa",
            str(FIXTURES / "nasa"),
            "--root",
            str(root / "lake"),
            "--workers",
            "1",
        ]
    )
    assert rc == 0
    return root / "lake"
