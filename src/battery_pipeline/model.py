"""Early-cycle life prediction: the paper's elastic-net baselines and candidates.

Protocol (pre-registered in docs/experiments.md):

* target is log10(cycle life); predictions are converted back to cycles;
* features are standardised on the training set only;
* elastic-net hyper-parameters (alpha, l1_ratio) come from cross-validation
  on the 41 training cells (4-fold, repeated 5 times; the paper used 4-fold CV
  with Monte Carlo sampling);
* an "improvement" candidate is chosen ONLY by repeated 4-fold CV RMSE on
  the training cells, before its test scores are looked at; all candidates'
  test scores are reported anyway.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from battery_pipeline import features as F

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

L1_RATIOS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
SEED = 0

#: Published results, Severson et al. 2019, model-metrics table (Table 2 of the
#: LBL accepted manuscript, escholarship.org/uc/item/9532z8t0). test1_excl
#: excludes the one primary-test cell that "reaches 80% SOH rapidly".
PAPER = {
    "variance": {
        "rmse": {"train": 103, "test1": 138, "test1_excl": 138, "test2": 196},
        "mpe": {"train": 14.1, "test1": 14.7, "test1_excl": 13.2, "test2": 11.4},
    },
    "discharge": {
        "rmse": {"train": 76, "test1": 91, "test1_excl": 86, "test2": 173},
        "mpe": {"train": 9.8, "test1": 13.0, "test1_excl": 10.1, "test2": 8.6},
    },
    "full": {
        "rmse": {"train": 51, "test1": 118, "test1_excl": 100, "test2": 214},
        "mpe": {"train": 5.6, "test1": 14.1, "test1_excl": 7.5, "test2": 10.7},
    },
}


def elastic_net() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    l1_ratio=L1_RATIOS,
                    alphas=100,
                    cv=RepeatedKFold(n_splits=4, n_repeats=5, random_state=SEED),
                    max_iter=200_000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def gradient_boosting() -> Pipeline:
    # fixed, conservative settings: 41 training cells leave no room for tuning
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED
                ),
            ),
        ]
    )


@dataclass
class Spec:
    name: str
    features: list[str]
    make: Callable[[], Pipeline]
    role: str  # "baseline" | "candidate"
    description: str = ""


SPECS = [
    Spec("variance", F.VARIANCE_FEATURES, elastic_net, "baseline", "log10 var ΔQ100-10(V) only"),
    Spec("discharge", F.DISCHARGE_FEATURES, elastic_net, "baseline", "6 discharge features"),
    Spec("full", F.FULL_FEATURES, elastic_net, "baseline", "9 features incl. charge time, temperature, IR"),
    Spec(
        "gbr_candidates",
        F.CANDIDATE_FEATURES,
        gradient_boosting,
        "candidate",
        "gradient boosting on all 20 candidate features",
    ),
    Spec(
        "enet_candidates_temp",
        F.CANDIDATE_FEATURES + F.TEMPERATURE_EXTRA,
        elastic_net,
        "candidate",
        "elastic net on 20 candidates + ΔT100-10(V) and Tmax-trend features",
    ),
    Spec(
        "enet_discharge_ica",
        F.DISCHARGE_FEATURES + F.ICA_EXTRA,
        elastic_net,
        "candidate",
        "discharge model + cellpy dQ/dV peak features",
    ),
]


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mpe(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p) / y) * 100)


def _sized(pipe: Pipeline, n: int) -> Pipeline:
    """Shrink the inner CV for tiny samples (the committed fixture); no-op for >= 8 cells."""
    est = pipe.named_steps["model"]
    if isinstance(est, ElasticNetCV) and n < 8:
        est.cv = RepeatedKFold(n_splits=max(2, min(4, n)), n_repeats=5, random_state=SEED)
    return pipe


def cv_rmse(make: Callable[[], Pipeline], x: np.ndarray, y_log: np.ndarray, n_repeats: int = 5) -> float:
    """Repeated 4-fold CV RMSE in cycles on the training cells."""
    errs = []
    k = max(2, min(4, len(x)))
    for tr, va in RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=SEED + 1).split(x):
        m = _sized(clone(make()), len(tr)).fit(x[tr], y_log[tr])
        errs.append(rmse(10 ** y_log[va], 10 ** m.predict(x[va])))
    return float(np.mean(errs))


@dataclass
class Result:
    spec: Spec
    pipeline: Pipeline
    n: dict[str, int]
    rmse: dict[str, float]
    mpe: dict[str, float]
    cv_rmse: float
    predictions: pd.DataFrame
    params: dict = field(default_factory=dict)
    x_train: np.ndarray | None = None


def fit_spec(spec: Spec, data: pd.DataFrame, outlier: str | None) -> Result:
    """Fit on split == train, score every split. ``data`` has features + split + cycle_life."""
    use = data.dropna(subset=spec.features + ["cycle_life"])
    train = use[use["split"] == "train"]
    x_tr = train[spec.features].to_numpy(float)
    y_tr = np.log10(train["cycle_life"].to_numpy(float))
    pipe = _sized(spec.make(), len(x_tr)).fit(x_tr, y_tr)
    pred = use[["cell_id", "split", "cycle_life"]].copy()
    pred["predicted"] = 10 ** pipe.predict(use[spec.features].to_numpy(float))
    pred["model"] = spec.name

    scores_r, scores_p, n = {}, {}, {}
    groups = {
        "train": pred["split"] == "train",
        "test1": pred["split"] == "test1",
        "test1_excl": (pred["split"] == "test1") & (pred["cell_id"] != outlier),
        "test2": pred["split"] == "test2",
    }
    for key, mask in groups.items():
        sub = pred[mask]
        n[key] = len(sub)
        if len(sub):
            scores_r[key] = rmse(sub["cycle_life"].to_numpy(), sub["predicted"].to_numpy())
            scores_p[key] = mpe(sub["cycle_life"].to_numpy(), sub["predicted"].to_numpy())

    params: dict = {"n_features": len(spec.features)}
    est = pipe.named_steps["model"]
    if isinstance(est, ElasticNetCV):
        params.update(
            alpha=float(est.alpha_), l1_ratio=float(est.l1_ratio_), n_selected=int(np.sum(est.coef_ != 0))
        )
        params["coefficients"] = {f: float(c) for f, c in zip(spec.features, est.coef_, strict=True)}
    return Result(spec, pipe, n, scores_r, scores_p, cv_rmse(spec.make, x_tr, y_tr), pred, params, x_tr)


def run(
    data: pd.DataFrame,
    outlier: str | None,
    tracking_dir: Path,
    log: Callable[[str], None] = print,
    register: bool = True,
    data_noqc: pd.DataFrame | None = None,
) -> dict:
    """Fit every spec, log runs to MLflow, register baseline + selected candidate.

    ``data_noqc`` (features computed without dropping QC-flagged cycles) is
    used to refit the three baselines, to show what the cleaning changes.
    """
    results = [fit_spec(s, data, outlier) for s in SPECS]
    if data_noqc is not None:
        for s in SPECS:
            if s.role == "baseline":
                noqc = Spec(
                    f"{s.name}_noqc",
                    s.features,
                    s.make,
                    "baseline_noqc",
                    f"{s.description}; features without QC cleaning",
                )
                results.append(fit_spec(noqc, data_noqc, outlier))
    candidates = [r for r in results if r.spec.role == "candidate"]
    selected = min(candidates, key=lambda r: r.cv_rmse)
    best_baseline = min((r for r in results if r.spec.role == "baseline"), key=lambda r: r.cv_rmse)
    for r in results:
        log(
            f"  {r.spec.name:22s} cv={r.cv_rmse:6.1f}  "
            + "  ".join(f"{k}={v:6.1f}" for k, v in r.rmse.items())
        )

    _log_mlflow(results, selected, best_baseline, tracking_dir, register)

    return {
        "outlier_cell": outlier,
        "selection_rule": "lowest repeated 4-fold CV RMSE on the training cells, among candidates",
        "selected_candidate": selected.spec.name,
        "best_baseline_by_cv": best_baseline.spec.name,
        "models": {
            r.spec.name: {
                "role": r.spec.role,
                "description": r.spec.description,
                "features": r.spec.features,
                "n": r.n,
                "rmse": r.rmse,
                "mpe": r.mpe,
                "cv_rmse_train": r.cv_rmse,
                "params": r.params,
                "paper": PAPER.get(r.spec.name.removesuffix("_noqc")),
            }
            for r in results
        },
        "predictions": pd.concat([r.predictions for r in results], ignore_index=True),
    }


def _log_mlflow(
    results: list[Result], selected: Result, baseline: Result, tracking_dir: Path, register: bool
) -> None:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(tracking_dir / 'mlflow.db').as_posix()}")
    name = "cycle-life-prediction"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(tracking_dir / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(name)
    for r in results:
        with mlflow.start_run(run_name=r.spec.name):
            mlflow.set_tags({"role": r.spec.role, "dataset": "MATR Severson 2019", "split": "paper"})
            mlflow.log_params(
                {
                    "features": ",".join(r.spec.features),
                    "estimator": type(r.pipeline[-1]).__name__,
                    **{k: v for k, v in r.params.items() if not isinstance(v, dict)},
                }
            )
            mlflow.log_metrics({f"rmse_{k}": v for k, v in r.rmse.items()})
            mlflow.log_metrics({f"mpe_{k}": v for k, v in r.mpe.items()})
            mlflow.log_metric("cv_rmse_train", r.cv_rmse)
            mlflow.log_dict(r.params, "params.json")
            mlflow.log_text(r.predictions.to_csv(index=False), "predictions.csv")
            mlflow.log_figure(_parity_figure(r), "parity.png")
            reg_name = None
            if register and r is selected:
                reg_name = "battery-cycle-life-candidate"
            elif register and r is baseline:
                reg_name = "battery-cycle-life-baseline"
            mlflow.sklearn.log_model(
                r.pipeline,
                name="model",
                signature=infer_signature(r.x_train, r.pipeline.predict(r.x_train)),
                input_example=r.x_train[:2],
                registered_model_name=reg_name,
                # the model was fitted in this process, so its own sklearn types are trusted
                skops_trusted_types=trusted_types(r.pipeline),
            )


def trusted_types(pipeline: Pipeline) -> list[str]:
    """Types skops would refuse to load for this (self-made) pipeline."""
    import skops.io as sio

    return sorted(sio.get_untrusted_types(data=sio.dumps(pipeline)))


def _parity_figure(r: Result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    colors = {"train": "#1f77b4", "test1": "#d62728", "test2": "#ff7f0e"}
    for split, g in r.predictions.groupby("split"):
        ax.scatter(g["cycle_life"], g["predicted"], s=18, label=split, color=colors.get(split))
    lim = [0, max(r.predictions["cycle_life"].max(), r.predictions["predicted"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set(xlim=lim, ylim=lim, xlabel="observed cycle life", ylabel="predicted cycle life", title=r.spec.name)
    ax.legend()
    fig.tight_layout()
    return fig


def results_json(results: dict) -> str:
    return json.dumps({k: v for k, v in results.items() if k != "predictions"}, indent=2, sort_keys=False)
