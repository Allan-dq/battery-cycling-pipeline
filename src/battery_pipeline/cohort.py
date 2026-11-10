"""The paper's cell cohort and train / primary-test / secondary-test split.

Taken from the authors' published loading code ("Load Data" notebook in
github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation):
cells that did not reach 80% capacity in batch 1 and noisy channels in batch 3
are removed, the five continued batch-1 cells are merged (done at ingest),
cells are ordered by batch then index, and the first 84 (batches 1-2) are split
alternately into test (even positions, plus position 83) and train (odd).
"""

from __future__ import annotations

import pandas as pd

PAPER_REMOVED = {
    "b1c8": "did not reach 80% capacity",
    "b1c10": "did not reach 80% capacity",
    "b1c12": "did not reach 80% capacity",
    "b1c13": "did not reach 80% capacity",
    "b1c22": "did not reach 80% capacity",
    "b3c2": "noisy channel",
    "b3c23": "noisy channel",
    "b3c32": "noisy channel",
    "b3c37": "noisy channel",
    "b3c42": "noisy channel",
    "b3c43": "noisy channel",
}


def _order_key(cell_id: str) -> tuple[int, int]:
    b, c = cell_id[1:].split("c")
    return int(b), int(c)


def paper_split(cells: pd.DataFrame) -> pd.DataFrame:
    """Return ``cell_id, split`` for the paper cohort (train / test1 / test2).

    ``test1`` is the primary test set, ``test2`` the secondary one. Raises if
    the cohort does not have the paper's 41 + 43 + 40 cells, which would mean
    the raw files differ from the published ones.
    """
    matr = cells.loc[cells["source"] == "matr", "cell_id"] if "source" in cells else cells["cell_id"]
    ids = sorted((c for c in matr if c not in PAPER_REMOVED), key=_order_key)
    b12 = [c for c in ids if _order_key(c)[0] in (1, 2)]
    b3 = [c for c in ids if _order_key(c)[0] == 3]
    n12 = len(b12)
    test_pos = set(range(0, n12, 2)) | {n12 - 1}
    split = {c: ("test1" if i in test_pos else "train") for i, c in enumerate(b12)}
    split.update({c: "test2" for c in b3})
    out = pd.DataFrame({"cell_id": list(split), "split": list(split.values())})
    counts = out["split"].value_counts().to_dict()
    expected = {"train": 41, "test1": 43, "test2": 40}
    if counts != expected and len(matr) >= 135:
        raise ValueError(f"cohort sizes {counts} differ from the paper's {expected}")
    return out
