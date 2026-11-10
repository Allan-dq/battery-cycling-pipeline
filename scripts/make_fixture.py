"""Build the small committed test fixture from the real MATR files.

    python scripts/make_fixture.py --raw data/raw/matr --out tests/fixtures

Writes one HDF5 file per batch with the same MATLAB-v7.3 layout as the
originals (``batch`` group of object references), but only a few cells, the
first ~105 cycles and fewer samples:

* time series: every sample where the current changes by more than
  ``--keep-di`` A, its neighbours, and every ``--every``-th sample otherwise,
  so coulomb counting on the thinned data stays within a fraction of a percent;
* Q(V)/T(V) curves: every ``--grid-step``-th voltage point.

The data are real (CC BY 4.0, Severson et al. 2019, data.matr.io); only the
amount is reduced. ``cell_index`` keeps each cell's original position so cell
ids match the full dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

CELLS = {
    "2017-05-12": ("b1", {0: 60, 1: 105, 2: 105, 3: 105, 4: 105, 5: 105}),
    "2017-06-30": ("b2", {7: 50, 0: 105, 1: 105, 2: 105, 3: 105}),
    "2018-04-12": ("b3", {0: 105, 1: 105, 3: 105}),
}
CYCLE_FIELDS = ["t", "I", "V", "T", "Qc", "Qd", "Qdlin", "Tdlin"]
SUMMARY_FIELDS = ["IR", "QCharge", "QDischarge", "Tavg", "Tmin", "Tmax", "chargetime", "cycle"]


def thin(current: np.ndarray, every: int, keep_di: float) -> np.ndarray:
    n = len(current)
    if n <= 3:
        return np.arange(n)
    jump = np.abs(np.diff(current)) > keep_di
    keep = np.zeros(n, dtype=bool)
    keep[::every] = True
    keep[0] = keep[-1] = True
    keep[1:][jump] = True
    keep[:-1][jump] = True
    return np.flatnonzero(keep)


class Writer:
    def __init__(self, f: h5py.File):
        self.f = f
        self.refs = f.create_group("#refs#")
        self.n = 0

    def data(self, array, matlab_class: str = "double") -> h5py.Reference:
        self.n += 1
        ds = self.refs.create_dataset(f"d{self.n}", data=array, compression="gzip", compression_opts=9, shuffle=True)
        ds.attrs["MATLAB_class"] = np.bytes_(matlab_class)
        return ds.ref

    def group(self) -> h5py.Group:
        self.n += 1
        return self.refs.create_group(f"g{self.n}")


def build(src: Path, dst: Path, cells: dict[int, int], every: int, keep_di: float, grid_step: int) -> None:
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        b = fin["batch"]
        w = Writer(fout)
        out = fout.create_group("batch")
        idx = list(cells)
        names = ["summary", "cycles", "cycle_life", "policy_readable", "Vdlin"]
        refs = {k: [] for k in names}
        for i in idx:
            n_cyc = cells[i]
            refs["cycle_life"].append(w.data(fin[b["cycle_life"][i, 0]][()]))
            refs["policy_readable"].append(w.data(fin[b["policy_readable"][i, 0]][()], "char"))
            refs["Vdlin"].append(w.data(np.asarray(fin[b["Vdlin"][i, 0]][()])[:, ::grid_step]))
            s_in = fin[b["summary"][i, 0]]
            s_out = w.group()
            for k in SUMMARY_FIELDS:
                s_out.create_dataset(k, data=np.asarray(s_in[k][()])[:, :n_cyc])
            refs["summary"].append(s_out.ref)
            c_in = fin[b["cycles"][i, 0]]
            c_out = w.group()
            per_field = {k: [] for k in CYCLE_FIELDS}
            for j in range(n_cyc):
                current = np.asarray(fin[c_in["I"][j, 0]][()]).ravel()
                keep = thin(current * 1.1, every, keep_di)
                for k in CYCLE_FIELDS:
                    arr = np.asarray(fin[c_in[k][j, 0]][()])
                    flat = arr.ravel()
                    if k in ("Qdlin", "Tdlin"):
                        flat = flat[::grid_step] if flat.size > 10 else flat
                    elif flat.size == current.size:
                        flat = flat[keep]
                    shape = (flat.size, 1) if arr.shape[1] == 1 else (1, flat.size)
                    per_field[k].append(w.data(flat.reshape(shape)))
            for k in CYCLE_FIELDS:
                c_out.create_dataset(k, data=np.array(per_field[k], dtype=h5py.ref_dtype).reshape(-1, 1))
            refs["cycles"].append(c_out.ref)
        for k, v in refs.items():
            out.create_dataset(k, data=np.array(v, dtype=h5py.ref_dtype).reshape(-1, 1))
        out.create_dataset("cell_index", data=np.array(idx, dtype=np.int32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/matr"))
    ap.add_argument("--out", type=Path, default=Path("tests/fixtures"))
    ap.add_argument("--every", type=int, default=12)
    ap.add_argument("--keep-di", type=float, default=0.05)
    ap.add_argument("--grid-step", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for prefix, (label, cells) in CELLS.items():
        src = next(args.raw.glob(f"{prefix}_*.mat"))
        dst = args.out / f"matr_sample_{label}.h5"
        build(src, dst, cells, args.every, args.keep_di, args.grid_step)
        print(f"{dst}  {dst.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
