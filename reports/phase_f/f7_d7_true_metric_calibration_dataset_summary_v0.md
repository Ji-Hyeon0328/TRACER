# TRACER Phase-F7 D7 True-Metric Calibration Dataset v0

This dataset attaches Phase-F6 true-metric beta target seeds to D7 runtime shadow logs from Phase-F3 rollouts.

- input F6: `datasets/phase_f/f6_true_metric_beta_targets_v0.csv`
- output csv: `datasets/phase_f/f7_d7_true_metric_calibration_dataset_v0.csv`
- total rows: `6677`
- missing sources: `0`

## Rows by source condition

| tag | d7_rows | output_rows | beta_motion | beta_stability | beta_energy |
|---|---:|---:|---:|---:|---:|
| clean | 2226 | 2226 | 0.3333 | 0.3333 | 0.3333 |
| m030 | 2225 | 2225 | 0.0441 | 0.3104 | 0.6456 |
| p030 | 2226 | 2226 | 0.6549 | 0.2985 | 0.0466 |

## Safe interpretation

This is a calibration dataset, not a final IRL Objective Selector dataset. Rows are timestep-expanded from only three rollout conditions, so they are useful for wiring/calibration and sanity checks, not for strong generalization claims.
