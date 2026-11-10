# Pre-registered modelling protocol

Written 2026-11-09, before any test-set score was looked at, and committed
before the first full run finished. Amend by adding a dated section at the
end; do not edit the rules above it.

## Data and split

- Cohort and split exactly as the authors' published loading code: 124 cells,
  41 train / 43 primary test / 40 secondary test (`src/battery_pipeline/cohort.py`).
- Target: log10 of the dataset-reported cycle life (cycles to 0.88 Ah), with
  the five continued batch-1 cells extended as in the authors' code.
- "Primary test excl." drops the one primary-test cell that reaches 80% the
  fastest, which is what the paper's parenthetical numbers do.
- Metrics: RMSE in cycles and mean absolute percent error, on train, primary
  test (with and without that cell) and secondary test.

## Baselines (reproduction)

The paper's three elastic-net models with the paper's feature sets
(`variance`, `discharge`, `full`), standardised on the training cells, with
`alpha` and `l1_ratio` chosen by 4-fold cross-validation repeated 5 times on
the training cells. Features are computed after dropping QC-flagged broken
cycles (`docs/outlier_criteria.md`); the same three models are also refit on
features without that cleaning (`*_noqc`) and both are reported.

## Improvement candidates

Fixed before looking at any test score:

1. `gbr_candidates` — gradient boosting (300 trees, depth 2, learning rate
   0.05, subsample 0.8; no tuning) on all 20 candidate features.
2. `enet_candidates_temp` — elastic net on the 20 candidates plus three
   temperature features: log10 var of ΔT_{100-10}(V), mean ΔT_{100-10}(V),
   slope of per-cycle max temperature over cycles 2-100.
3. `enet_discharge_ica` — the `discharge` model plus two dQ/dV features from
   cellpy's incremental-capacity analysis (peak-height ratio and peak-voltage
   shift between cycles 10 and 100).

Selection rule: the candidate with the lowest repeated 4-fold CV RMSE on the
41 training cells is "the improvement". Its test scores are reported next to
the baselines whatever they are, and so are the other two candidates'. If it
does not beat the baselines on the test sets, the README says so.
