"""Paste reports/results.md into README.md between the results markers.

    python scripts/update_readme.py

So the README never carries a number that `bcp report` did not produce.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- results:start -->", "<!-- results:end -->"


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    block = (ROOT / "reports" / "results.md").read_text().strip()
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{block}\n{END}{tail}")
    print(f"updated {readme}")


if __name__ == "__main__":
    main()
