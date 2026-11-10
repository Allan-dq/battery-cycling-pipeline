"""Command line: ``bcp <stage>`` or ``bcp all``.

bcp all                                                   # full run on the downloaded files
bcp all --raw tests/fixtures --nasa tests/fixtures/nasa --root out/sample   # committed sample
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from battery_pipeline import process, report, telemetry, train
from battery_pipeline.ingest import ingest_matr, ingest_nasa
from battery_pipeline.store import Lake

DEFAULT_ROOT = Path("data/processed")


def _raw_files(raw: Path) -> list[Path]:
    files = sorted(p for p in raw.iterdir() if p.suffix in {".mat", ".h5"})
    if not files:
        raise SystemExit(f"no .mat/.h5 files in {raw}; run scripts/download_data.py first")
    return files


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="bcp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=["ingest", "process", "warehouse", "train", "report", "all"])
    ap.add_argument("--raw", type=Path, default=Path("data/raw/matr"), help="folder with MATR .mat files")
    ap.add_argument(
        "--nasa",
        type=Path,
        default=Path("data/raw/nasa/extracted"),
        help="folder searched for NASA PCoE B00xx.mat files (skipped if absent)",
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="lake / warehouse folder")
    ap.add_argument(
        "--reports", type=Path, default=None, help="where results.json/.md go (default: <root>/reports)"
    )
    ap.add_argument("--tracking", type=Path, default=None, help="MLflow store folder (default: <root>)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-register", action="store_true", help="skip MLflow model registration")
    args = ap.parse_args(argv)

    lake = Lake(args.root)
    reports = args.reports or args.root / "reports"
    tracking = args.tracking or args.root
    stages = ["ingest", "process", "warehouse", "train", "report"] if args.stage == "all" else [args.stage]
    t0 = time.perf_counter()
    for st in stages:
        _log(f"stage {st}")
        if st == "ingest":
            r = ingest_matr(_raw_files(args.raw), lake, log=_log)
            _log(f"  {r.cells} cells, {r.cycles:,} cycles, {r.rows:,} rows; merged {r.merged_continuations}")
            if args.nasa.exists() and any(args.nasa.rglob("B00*.mat")):
                r = ingest_nasa(args.nasa, lake, log=_log)
                _log(f"  NASA: {r.cells} cells, {r.cycles:,} cycles, {r.rows:,} rows")
        elif st == "process":
            r = process.run(lake, workers=args.workers, log=_log)
            _log(f"  {r}")
        elif st == "warehouse":
            _log(f"  {lake.build_warehouse()}")
        elif st == "train":
            train.run(lake, tracking, log=_log, register=not args.no_register)
        elif st == "report":
            lake.build_warehouse()
            report.write(lake, reports)
            _log(f"  wrote {reports / 'results.md'}")
    telemetry.write(args.root / "pipeline_metrics.prom")
    _log(f"done in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
