"""Download the public raw datasets listed in data/manifest.json.

Usage:
    python scripts/download_data.py                 # every dataset
    python scripts/download_data.py --dataset matr  # one dataset
    python scripts/download_data.py --verify-only   # check sizes/checksums

Files land in data/raw/<dataset>/ (git-ignored). Downloads resume from a
``.part`` file. A file is accepted only if its size and SHA-256 match the
manifest; a manifest entry without a checksum yet gets it filled in
(``--record``) so later downloads can be verified against it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

try:  # python.org builds ship without system CA certs; certifi fixes that
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    SSL_CONTEXT = ssl.create_default_context()

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"
RAW = ROOT / "data" / "raw"
CHUNK = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, size: int | None, retries: int = 8) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have == size:
            break
        req = urllib.request.Request(url, headers={"User-Agent": "battery-cycling-pipeline/0.1"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as resp, part.open("ab" if have else "wb") as out:
                if have and resp.status != 206:  # server ignored Range: start over
                    out.seek(0)
                    out.truncate()
                last = time.time()
                while block := resp.read(CHUNK):
                    out.write(block)
                    if time.time() - last > 30:
                        print(f"  {dest.name}: {out.tell() / 1e9:.2f} GB", flush=True)
                        last = time.time()
            break
        except OSError as exc:  # network hiccup: resume
            if isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
                raise  # not transient
            print(f"  attempt {attempt} failed: {exc}; retrying", flush=True)
            time.sleep(min(60, 5 * attempt))
    else:
        raise RuntimeError(f"could not download {url}")
    part.rename(dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", help="dataset key in the manifest (repeatable)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--record", action="store_true", help="write missing checksums into the manifest")
    args = ap.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text())
    keys = args.dataset or list(manifest["datasets"])
    bad = 0
    for key in keys:
        ds = manifest["datasets"][key]
        folder = RAW / key
        folder.mkdir(parents=True, exist_ok=True)
        for f in ds["files"]:
            dest = folder / f["name"]
            if not dest.exists() and not args.verify_only:
                print(f"downloading {f['name']} ({f.get('bytes', 0) / 1e9:.2f} GB) from {f['url']}", flush=True)
                fetch(f["url"], dest, f.get("bytes"))
            if not dest.exists():
                print(f"MISSING  {dest.relative_to(ROOT)}")
                bad += 1
                continue
            size_ok = f.get("bytes") is None or dest.stat().st_size == f["bytes"]
            digest = sha256(dest)
            if f.get("sha256") is None:
                status = "NEW"
                if args.record:
                    f["sha256"] = digest
            else:
                status = "OK" if digest == f["sha256"] and size_ok else "BAD"
            if not size_ok:
                status = "BAD"
            bad += status == "BAD"
            print(f"{status:8} {dest.relative_to(ROOT)}  {digest}")
    if args.record:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
