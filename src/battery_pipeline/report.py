"""Stage 4: every number the README cites, recomputed from the lake.

Writes ``reports/results.json`` (machine-readable) and ``reports/results.md``
(the tables pasted into the README). Nothing in the README is typed by hand.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from battery_pipeline import quality
from battery_pipeline.cohort import PAPER_REMOVED, paper_split
from battery_pipeline.store import Lake


def _du_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*.parquet")) if path.exists() else 0


def compute(lake: Lake) -> dict:
    cells = lake.read_table("cells")
    cycles = lake.read_table("cycles")
    flags = lake.read_table("qc_cycle_flags")
    cell_flags = lake.read_table("qc_cell_flags")
    con = lake.connect()
    try:
        n_rows = int(con.execute("SELECT count(*) FROM timeseries").fetchone()[0])
    finally:
        con.close()

    real = cycles.merge(
        flags[["cell_id", "cycle_index", "empty_cycle", "bad_cycle"]], on=["cell_id", "cycle_index"]
    )
    real = real[~real["empty_cycle"]]
    real["group"] = np.where(real["cell_id"].str.startswith("nasa_"), "nasa", real["cell_id"].str[:2])
    vendor = real["vendor_discharge_capacity_ah"]
    ok = vendor > 0
    diff_mah = (real.loc[ok, "discharge_capacity_ah"] - vendor[ok]).abs() * 1e3
    rel = diff_mah / (vendor[ok] * 1e3)
    signed = (real["discharge_capacity_ah"] - vendor) / vendor.where(ok)
    by_group = {}
    for grp, g in real.groupby("group"):
        sg = signed[g.index]
        by_group[grp] = {
            "cells": int(g["cell_id"].nunique()),
            "cycles": int(len(g)),
            "capacity_offset_pct_median": float(sg.median() * 100),
            "capacity_offset_pct_p10": float(sg.quantile(0.1) * 100),
            "capacity_offset_pct_p90": float(sg.quantile(0.9) * 100),
            "bad_cycle_pct": float(g["bad_cycle"].mean() * 100),
            # what an absolute 2% rule (the first version of capacity_mismatch) would flag
            "fixed_2pct_rule_flag_pct": float((sg.abs() > 0.02).mean() * 100),
            "coulombic_efficiency_median": float(g["coulombic_efficiency"].median()),
        }

    split = paper_split(cells)
    both = cells[cells["source"] == "matr"].dropna(subset=["cycle_life", "reported_cycle_life"])
    life_diff = (both["cycle_life"] - both["reported_cycle_life"]).abs()

    flags["group"] = np.where(flags["cell_id"].str.startswith("nasa_"), "nasa", "matr")
    flag_counts = {f: {g: int(v[f].sum()) for g, v in flags.groupby("group")} for f in quality.CYCLE_FLAGS}
    rule_cols = [c for c in cell_flags.columns if c.startswith("flag_")]
    flagged_cells = {c: sorted(cell_flags.loc[cell_flags[c], "cell_id"]) for c in rule_cols}
    anomalous = set(cell_flags.loc[cell_flags["anomalous"], "cell_id"])

    def _ids(col: str, source: str) -> list[str]:
        sel = cell_flags[cell_flags[col] & (cell_flags["source"] == source)]
        return sorted(sel["cell_id"])

    paper_noisy = {c for c, why in PAPER_REMOVED.items() if why == "noisy channel"}
    paper_no_eol = {c for c, why in PAPER_REMOVED.items() if why != "noisy channel"}
    matr_cells = cells[cells["source"] == "matr"]
    our_no_eol = set(matr_cells.loc[matr_cells["cycle_life"].isna(), "cell_id"])

    out = {
        "dataset": {
            "cells": int(len(cells)),
            "cycles": int(len(cycles)),
            "non_empty_cycles": int(len(real)),
            "timeseries_rows": n_rows,
            "cells_by_source": cells["source"].value_counts().to_dict(),
            "timeseries_parquet_bytes": _du_bytes(lake.timeseries),
            "curves_parquet_bytes": _du_bytes(lake.curves),
            "paper_cohort": split["split"].value_counts().to_dict(),
        },
        "metrics_validation": {
            "cycles_compared": int(ok.sum()),
            "discharge_capacity_abs_diff_mah_median": float(diff_mah.median()) if len(diff_mah) else None,
            "discharge_capacity_abs_diff_mah_p99": float(diff_mah.quantile(0.99)) if len(diff_mah) else None,
            "discharge_capacity_within_1pct": float((rel <= 0.01).mean()) if len(rel) else None,
            "coulombic_efficiency_median": float(real["coulombic_efficiency"].median()),
            "energy_efficiency_median": float(real["energy_efficiency"].median()),
            "dcir_proxy_mohm_median": float(real["dcir_proxy_ohm"].median() * 1e3),
            "dataset_ir_mohm_median": float(real["ir_ohm"].where(real["ir_ohm"] > 0).median() * 1e3),
            "by_group": by_group,
        },
        "cycle_life_check": {
            "cells_compared": int(len(both)),
            "exact_match": int((life_diff == 0).sum()),
            "within_1_cycle": int((life_diff <= 1).sum()),
            "max_abs_diff": float(life_diff.max()) if len(life_diff) else None,
        },
        "qc": {
            "rules": quality.rules_dict(),
            "cycle_flag_counts": flag_counts,
            "cycles_with_any_flag_excl_empty": int(flags.loc[~flags["empty_cycle"], "any_flag"].sum()),
            "cells_flagged_by_rule": flagged_cells,
            "cells_anomalous": {src: _ids("anomalous", src) for src in ("matr", "nasa")},
            "degradation_outliers": {src: _ids("degradation_outlier", src) for src in ("matr", "nasa")},
            "measurement_anomalies": {src: _ids("measurement_anomaly", src) for src in ("matr", "nasa")},
            "vs_paper": {
                "paper_noisy_batch3": sorted(paper_noisy),
                "paper_noisy_we_flag": sorted(paper_noisy & anomalous),
                "paper_noisy_never_reached_eol": sorted(paper_noisy & our_no_eol),
                "paper_no_eol_batch1": sorted(paper_no_eol),
                "paper_no_eol_we_find": sorted(paper_no_eol & our_no_eol),
                "we_find_no_eol_matr": sorted(our_no_eol),
            },
        },
    }
    results_path = lake.root / "model_results.json"
    if results_path.exists():
        out["models"] = json.loads(results_path.read_text())
        out["qc_effect"] = qc_effect(lake, out["models"]["models"])
    return out


def _fmt(x, nd=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def markdown(r: dict) -> str:
    d, mv, q = r["dataset"], r["metrics_validation"], r["qc"]
    lines = [
        "<!-- generated by `bcp report`; do not edit by hand -->",
        "### Data",
        "",
        f"- {d['cells']} cells ({d['cells_by_source'].get('matr', 0)} MATR after merging the five continued cells, "
        f"{d['cells_by_source'].get('nasa', 0)} NASA), {d['non_empty_cycles']:,} cycles with data, "
        f"{d['timeseries_rows']:,} time-series rows "
        f"({d['timeseries_parquet_bytes'] / 1e9:.2f} GB Parquet + {d['curves_parquet_bytes'] / 1e9:.2f} GB Q(V) curves).",
        f"- Paper cohort: {d['paper_cohort'].get('train', 0)} train / {d['paper_cohort'].get('test1', 0)} primary test / "
        f"{d['paper_cohort'].get('test2', 0)} secondary test.",
        "",
        "### Per-cycle metrics vs. the dataset's own summary",
        "",
        f"- Discharge capacity from coulomb counting vs. dataset value over {mv['cycles_compared']:,} cycles: "
        f"median |diff| {_fmt(mv['discharge_capacity_abs_diff_mah_median'], 2)} mAh, "
        f"99th percentile {_fmt(mv['discharge_capacity_abs_diff_mah_p99'], 1)} mAh, "
        f"{_fmt(100 * mv['discharge_capacity_within_1pct'], 1)}% of cycles within 1%.",
        f"- Median coulombic efficiency {_fmt(mv['coulombic_efficiency_median'], 4)}, "
        f"median energy efficiency {_fmt(mv['energy_efficiency_median'], 3)}.",
        f"- Median DCIR proxy {_fmt(mv['dcir_proxy_mohm_median'], 1)} mΩ vs. dataset pulse IR "
        f"{_fmt(mv['dataset_ir_mohm_median'], 1)} mΩ (MATR).",
        "",
        "| Group | Cells | Cycles | Coulomb counting vs. dataset capacity, median (p10–p90) | Cycles a fixed 2% rule would flag | Bad cycles (offset-relative rule) | Median CE |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for grp, g in mv["by_group"].items():
        lines.append(
            f"| {grp} | {g['cells']} | {g['cycles']:,} | {g['capacity_offset_pct_median']:+.2f}% "
            f"({g['capacity_offset_pct_p10']:+.2f}% to {g['capacity_offset_pct_p90']:+.2f}%) | "
            f"{g['fixed_2pct_rule_flag_pct']:.1f}% | {g['bad_cycle_pct']:.1f}% | {g['coulombic_efficiency_median']:.4f} |"
        )
    lines += [
        "",
        "### Data quality",
        "",
        "| Cycle check | MATR cycles flagged | NASA cycles flagged |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| `{k}` | {v.get('matr', 0):,} | {v.get('nasa', 0):,} |" for k, v in q["cycle_flag_counts"].items()
    ]
    vp = q["vs_paper"]
    lines += [
        "",
        "| Cell-level result | MATR | NASA |",
        "|---|---|---|",
        "| Degradation outliers (early fade, \\|z\\| > 3.5) | "
        + " | ".join(
            f"{len(q['degradation_outliers'][s])}: {', '.join(q['degradation_outliers'][s]) or '–'}"
            for s in ("matr", "nasa")
        )
        + " |",
        "| Measurement anomalies (temperature, IR, CE, sensor, isolation forest) | "
        + " | ".join(
            f"{len(q['measurement_anomalies'][s])}: {', '.join(q['measurement_anomalies'][s]) or '–'}"
            for s in ("matr", "nasa")
        )
        + " |",
        "",
        f"- Cycle life: our end-of-life rule reproduces the dataset's reported cycle life exactly for "
        f"{r['cycle_life_check']['exact_match']} of {r['cycle_life_check']['cells_compared']} MATR cells "
        f"(max difference {_fmt(r['cycle_life_check']['max_abs_diff'])} cycles).",
        f"- Of the {len(vp['paper_no_eol_batch1'])} batch-1 cells the authors dropped for not reaching 80%, our rule "
        f"finds {len(vp['paper_no_eol_we_find'])} ({len(vp['we_find_no_eol_matr'])} MATR cells in total never reach 80%).",
        f"- Of the {len(vp['paper_noisy_batch3'])} batch-3 cells the authors' loader drops as noisy, "
        f"{len(vp['paper_noisy_never_reached_eol'])} never reached 80% ({', '.join(vp['paper_noisy_never_reached_eol']) or 'none'}); "
        f"the cell-level rules flag {len(vp['paper_noisy_we_flag'])} of the six"
        f"{(': ' + ', '.join(vp['paper_noisy_we_flag'])) if vp['paper_noisy_we_flag'] else ''}.",
        "",
    ]
    if "models" in r:
        m = r["models"]
        lines += [
            "### Early-cycle life prediction (paper split)",
            "",
            "RMSE in cycles; mean % error in brackets. *Primary test excl.* drops the one primary-test cell "
            f"that reaches 80% fastest (`{m['outlier_cell']}`), as the paper's parenthetical numbers do.",
            "",
            "| Model | Train | Primary test | Primary test excl. | Secondary test | Train CV RMSE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, v in m["models"].items():
            cells_ = [
                f"{_fmt(v['rmse'].get(k))} ({_fmt(v['mpe'].get(k), 1)}%)"
                for k in ("train", "test1", "test1_excl", "test2")
            ]
            lines.append(f"| {name} | " + " | ".join(cells_) + f" | {_fmt(v['cv_rmse_train'])} |")
            if v.get("paper"):
                p = v["paper"]
                pc = [f"{p['rmse'][k]} ({p['mpe'][k]}%)" for k in ("train", "test1", "test1_excl", "test2")]
                lines.append("| ↳ paper | " + " | ".join(pc) + " | |")
        lines += [
            "",
            f"Candidate chosen by training-set CV only: **{m['selected_candidate']}** "
            f"(best baseline by the same CV: {m['best_baseline_by_cv']}).",
            "",
        ]
        lines += findings(r)
    return "\n".join(lines)


SPLIT_NAMES = {
    "train": "train",
    "test1": "primary test",
    "test1_excl": "primary test excl.",
    "test2": "secondary test",
}


def _cycles(x: float, signed: bool = False) -> str:
    if not signed and abs(x) < 1:
        return "under 1 cycle"
    n = round(x)
    return f"{n:+d} cycle{'s' if abs(n) != 1 else ''}" if signed else f"{n:d} cycle{'s' if n != 1 else ''}"


def findings(r: dict) -> list[str]:
    """Plain-language comparisons, every number computed from the results."""
    m = r["models"]
    models = m["models"]
    out = ["**What the table says**", ""]
    for name in ("variance", "discharge", "full"):
        clean, raw = models.get(name), models.get(f"{name}_noqc")
        if not clean or not clean.get("paper") or not raw:
            continue
        paper = clean["paper"]["rmse"]

        def worst(res: dict, paper: dict = paper) -> tuple[float, str]:
            gaps = {k: res["rmse"][k] - paper[k] for k in SPLIT_NAMES if k in res["rmse"]}
            k = max(gaps, key=lambda key: abs(gaps[key]))
            return gaps[k], SPLIT_NAMES[k]

        (gc, kc), (gr, kr) = worst(clean), worst(raw)
        out.append(
            f"- `{name}`: largest RMSE gap to the paper is {_cycles(gc, signed=True)} ({kc}) with QC cleaning "
            f"and {_cycles(gr, signed=True)} ({kr}) without."
        )
    qe = r.get("qc_effect")
    if qe:
        out.append(
            f"- QC cleaning matters most for `{qe['model']}` on the primary test set "
            f"({qe['rmse_clean']:.0f} vs. {qe['rmse_raw']:.0f} cycles). Most of the difference is one cell, "
            f"`{qe['cell_id']}` (observed {qe['observed']:.0f} cycles): predicted {_cycles(qe['pred_raw'])} from "
            f"uncleaned features and {_cycles(qe['pred_clean'])} after its broken early cycle(s) are dropped ("
            + "; ".join(
                f"cycle {b['cycle']}: dataset capacity {b['dataset_capacity_ah']:.2f} Ah, coulomb-counted "
                f"{b['coulomb_counted_ah']:.2f} Ah over {b['duration_s'] / 3600:.1f} h"
                for b in qe["broken_cycles"]
            )
            + ")."
        )
    sel, base = models[m["selected_candidate"]], models[m["best_baseline_by_cv"]]
    verdict = []
    for k in ("test1_excl", "test2"):
        d = sel["rmse"][k] - base["rmse"][k]
        verdict.append(f"{SPLIT_NAMES[k]} {sel['rmse'][k]:.0f} vs. {base['rmse'][k]:.0f} ({d:+.0f})")
    better = all(sel["rmse"][k] < base["rmse"][k] for k in ("test1_excl", "test2"))
    out.append(
        f"- Selected candidate `{m['selected_candidate']}` vs. best baseline `{m['best_baseline_by_cv']}`: "
        + "; ".join(verdict)
        + (
            ". It is better on both test sets."
            if better
            else ". It does not beat the baseline on both test sets."
        )
    )
    n_full = models["full"]["n"]
    out.append(
        f"- `full` is scored on {n_full['train']}/{n_full['test1']}/{n_full['test2']} train/primary/secondary "
        f"cells; the paper dropped 4 cells from its full model because temperature probes lost contact, and "
        f"our temperature checks do not single those out."
    )
    out.append("")
    return out


def qc_effect(lake: Lake, models: dict) -> dict | None:
    """Which cell drives the biggest primary-test difference between cleaned and raw features."""
    path = lake.table("predictions")
    if not path.exists():
        return None
    pairs = [n for n in ("variance", "discharge", "full") if n in models and f"{n}_noqc" in models]
    if not pairs:
        return None
    name = max(pairs, key=lambda n: abs(models[n]["rmse"]["test1"] - models[f"{n}_noqc"]["rmse"]["test1"]))
    pred = pd.read_parquet(path)
    pred = pred[pred["split"] == "test1"]
    clean = pred[pred["model"] == name].set_index("cell_id")
    raw = pred[pred["model"] == f"{name}_noqc"].set_index("cell_id")
    err = (raw["predicted"] - raw["cycle_life"]).abs() - (clean["predicted"] - clean["cycle_life"]).abs()
    cell = str(err.idxmax())
    feats = pd.read_parquet(lake.table("features")).set_index("cell_id")
    flags = pd.read_parquet(lake.table("qc_cycle_flags"), filters=[("cell_id", "==", cell)])
    cyc = pd.read_parquet(lake.table("cycles"), filters=[("cell_id", "==", cell)])
    bad = flags[flags["bad_cycle"] & flags["cycle_index"].between(2, 100)][["cycle_index"]].merge(
        cyc, on="cycle_index"
    )
    return {
        "broken_cycles": [
            {
                "cycle": int(b.cycle_index),
                "dataset_capacity_ah": float(b.vendor_discharge_capacity_ah),
                "coulomb_counted_ah": float(b.discharge_capacity_ah),
                "duration_s": float(b.duration_s),
            }
            for b in bad.itertuples()
        ],
        "model": name,
        "rmse_clean": models[name]["rmse"]["test1"],
        "rmse_raw": models[f"{name}_noqc"]["rmse"]["test1"],
        "cell_id": cell,
        "observed": float(clean.at[cell, "cycle_life"]),
        "pred_clean": float(clean.at[cell, "predicted"]),
        "pred_raw": float(raw.at[cell, "predicted"]),
        "bad_cycles": int(feats.at[cell, "n_bad_cycles_2_100"]) if cell in feats.index else None,
    }


def write(lake: Lake, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = compute(lake)
    (out_dir / "results.json").write_text(json.dumps(r, indent=2, default=_json_default) + "\n")
    (out_dir / "results.md").write_text(markdown(r) + "\n")
    return r


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(type(o))


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""
