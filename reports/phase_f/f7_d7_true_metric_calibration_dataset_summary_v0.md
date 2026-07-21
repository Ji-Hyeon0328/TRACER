# TRACER Phase-F7 D7 True-Metric Calibration Dataset v0

This dataset attaches Phase-F6 true-metric beta target seeds to D7 runtime shadow logs from Phase-F3 rollouts.

- input F6: `datasets/phase_f/f6_true_metric_beta_targets_v0.csv`
- output csv: `datasets/phase_f/f7_d7_true_metric_calibration_dataset_v0.csv`
- total rows: `19814`
- missing sources: `0`

## Rows by source condition

| tag | d7_rows | output_rows | beta_motion | beta_stability | beta_energy |
|---|---:|---:|---:|---:|---:|
| clean | 2226 | 2226 | 0.3498 | 0.3003 | 0.3498 |
| clean_r1 | 2224 | 2224 | 0.2516 | 0.3673 | 0.3811 |
| clean_r2 | 2222 | 2222 | 0.3865 | 0.3758 | 0.2376 |
| m030 | 2225 | 2225 | 0.0414 | 0.3519 | 0.6067 |
| m030_r1 | 2122 | 2122 | 0.3986 | 0.1630 | 0.4383 |
| m030_r2 | 2225 | 2225 | 0.1182 | 0.4908 | 0.3910 |
| p030 | 2226 | 2226 | 0.5769 | 0.3820 | 0.0410 |
| p030_r1 | 2223 | 2223 | 0.2945 | 0.3787 | 0.3268 |
| p030_r2 | 2121 | 2121 | 0.3448 | 0.4635 | 0.1917 |

## Safe interpretation

This is a calibration dataset, not a final IRL Objective Selector dataset. Rows are timestep-expanded from only three rollout conditions, so they are useful for wiring/calibration and sanity checks, not for strong generalization claims.
