-- Reporting views over the warehouse tables. A view is created only when the
-- tables it reads exist, so a partial run still gets a usable warehouse.

-- one row per cell: life, policy and every cell-level QC result
CREATE VIEW cell_report AS
SELECT
    c.cell_id,
    c.batch,
    c.charge_policy,
    c.n_cycles,
    c.reported_cycle_life,
    c.cycle_life,
    q.fade_slope_mah_per_cycle,
    q.temperature_max_mean_c,
    q.temperature_residual_c,
    q.ir_mean_mohm,
    q.temp_fault_frac,
    q.iforest_score,
    q.n_flags,
    q.anomalous
FROM cells AS c
LEFT JOIN qc_cell_flags AS q USING (cell_id);

-- cycle-level QC rolled up per cell
CREATE VIEW cycle_quality AS
SELECT
    cell_id,
    count(*) AS cycles,
    sum(CAST(empty_cycle AS INTEGER)) AS empty_cycles,
    sum(CAST(time_gap AS INTEGER)) AS time_gap_cycles,
    sum(CAST(temperature_spike AS INTEGER)) AS temperature_spike_cycles,
    sum(CAST(temperature_missing AS INTEGER)) AS temperature_missing_cycles,
    sum(CAST(capacity_mismatch AS INTEGER)) AS capacity_mismatch_cycles,
    sum(CAST(any_flag AS INTEGER)) AS flagged_cycles
FROM qc_cycle_flags
GROUP BY cell_id;

-- capacity retention every 100 cycles, by charging policy
CREATE VIEW capacity_by_policy AS
SELECT
    c.charge_policy,
    y.cycle_index,
    count(*) AS cells,
    avg(y.discharge_capacity_ah) AS mean_discharge_capacity_ah,
    avg(y.temperature_max_c) AS mean_max_temperature_c
FROM cycles AS y
JOIN cells AS c USING (cell_id)
WHERE y.cycle_index % 100 = 0
GROUP BY c.charge_policy, y.cycle_index;

-- model errors by split, recomputed from stored predictions
CREATE VIEW prediction_errors AS
SELECT
    model,
    split,
    count(*) AS cells,
    sqrt(avg(power(predicted - cycle_life, 2))) AS rmse_cycles,
    100 * avg(abs(predicted - cycle_life) / cycle_life) AS mean_pct_error
FROM predictions
GROUP BY model, split;
