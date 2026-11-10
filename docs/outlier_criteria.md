# Data-quality and outlier-rejection criteria

Implemented in `src/battery_pipeline/quality.py`; the thresholds below are the
code's defaults (`CycleRules`, `RULES_BY_SOURCE`, `CellRules`). They come from
the cell datasheet / test protocol or from robust statistics. None of them
was chosen by looking at which cells the paper excluded; the comparison with
the paper's exclusions is a check, reported in `reports/results.md`.

## 1. Cycle checks (every cycle of every cell)

| Flag | Rule (MATR) | NASA difference | Why |
|---|---|---|---|
| `empty_cycle` | fewer than 10 samples | – | MATR cycle 1 is a two-sample placeholder |
| `duplicate_cycle` | same sample count and identical V/I fingerprint as the previous cycle | – | duplicated export |
| `time_gap` | largest step between samples > 60 s (12× the 5 s logging interval) | > 4 h (a NASA cycle includes a rest and an impedance sweep) | interrupted logging or paused test |
| `time_not_monotonic` | time goes backwards within the cycle | – | merged or corrupted records |
| `voltage_out_of_range` | V < 1.9 V or > 3.7 V (protocol 2.0–3.6 V ± 0.1 V) | 1.8–4.3 V | sensor glitch, wrong channel |
| `current_out_of_range` | \|I\| > 11 A (10C; fastest policy is 8C) | > 5 A (max load 4 A) | sensor glitch |
| `temperature_missing` | more than half the samples are NaN or exactly 0 °C | – | thermocouple not logged |
| `temperature_out_of_range` | T < 15 °C or > 60 °C (chamber at 30 °C) | −5 to 80 °C (ambient 4–44 °C) | thermocouple lost contact / spike |
| `temperature_spike` | \|ΔT\| > 3 °C between consecutive samples | – | contact noise |
| `capacity_mismatch` | coulomb-counted discharge capacity differs from the dataset's value by > 2% | – | dataset summary disagrees with its own raw data |
| `ce_out_of_range` | coulombic efficiency outside 0.95–1.05 | – | interrupted / partial cycle |

`bad_cycle` = `empty_cycle` or `time_not_monotonic` or `capacity_mismatch` or
`ce_out_of_range`. Bad cycles are dropped before features and cell indicators
are computed (if cycle 10 or 100 is bad, ΔQ(V) uses the nearest good cycle
within ±3). A `time_gap` alone does not make a cycle bad: capacity and the
voltage curve are still valid after a pause.

## 2. Anomalous cells (per source, cycles 2–100)

Indicators per cell, from good cycles only: capacity-fade slope (linear fit,
mAh/cycle), mean of per-cycle max temperature, pulse internal resistance,
median coulombic efficiency, fraction of cycles with a temperature fault,
fraction of bad cycles.

- **Robust z-score** (Iglewicz & Hoaglin 1993): z = 0.6745·(x − median)/MAD;
  a cell is flagged on an indicator when |z| > 3.5. Temperature is first
  corrected for test condition: residual of a linear fit on mean charge time
  (MATR, faster charging runs hotter) or on ambient temperature (NASA).
- **Temperature sensor**: flagged when > 5% of early cycles have a
  temperature fault (missing, out of range or spike).
- **Isolation forest** on the six indicators (robust-scaled, 500 trees,
  contamination 5%, seed 0) catches cells that are unusual in combination.

A cell is **anomalous** if any rule fires. Cells are compared only with cells
of the same source (test campaign).

## 3. What happens to anomalous cells

They stay in the lake and the warehouse (`cell_report` view shows why each
was flagged). The life-prediction benchmark keeps the paper's cohort so the
numbers stay comparable with the paper; QC affects it only through the
dropped bad cycles. A production pipeline would route anomalous cells to an
engineer's review queue rather than silently drop them.
