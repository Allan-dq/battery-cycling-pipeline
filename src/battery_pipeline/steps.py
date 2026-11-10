"""Split a cycle into charge / rest / discharge segments from the current."""

from __future__ import annotations

import numpy as np

#: |I| below this (A) counts as rest. The MATR CV steps end at C/50 = 22 mA.
REST_CURRENT_A = 0.005


def label_steps(
    current_a: np.ndarray, rest_current_a: float = REST_CURRENT_A
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(step_type, step_index)`` for each sample.

    ``step_type`` is "charge" (I > threshold), "discharge" (I < -threshold) or
    "rest". ``step_index`` starts at 1 and increases every time the type
    changes, so each contiguous segment gets its own index.
    """
    current = np.asarray(current_a, dtype=float)
    code = np.where(current > rest_current_a, 1, np.where(current < -rest_current_a, -1, 0))
    names = np.array(["discharge", "rest", "charge"], dtype=object)[code + 1]
    if len(code) == 0:
        return names, np.zeros(0, dtype=np.int16)
    changes = np.concatenate([[1], (np.diff(code) != 0).astype(int)])
    return names, np.cumsum(changes).astype(np.int16)
