# TRACER Phase-F19 D7 Robust-Beta Calibration Dataset v0

This attaches Phase-F18 robust median true-metric beta targets to D7 runtime shadow logs.

- input F18 rollout table: `datasets/phase_f/f18_rollout_with_robust_group_beta_targets_v0.csv`
- output csv: `datasets/phase_f/f19_d7_robust_beta_calibration_dataset_v0.csv`
- total rows: `45402`
- missing sources: `0`

## Rows by base condition

| base_tag | rollouts | output_rows | beta_motion | beta_stability | beta_energy |
|---|---:|---:|---:|---:|---:|
| clean | 3 | 6672 | 0.3807 | 0.2387 | 0.3807 |
| m015 | 3 | 5966 | 0.5301 | 0.2194 | 0.2505 |
| m030 | 3 | 6572 | 0.3610 | 0.1145 | 0.5245 |
| m060 | 3 | 6677 | 0.6204 | 0.1594 | 0.2202 |
| p015 | 3 | 6474 | 0.3885 | 0.2666 | 0.3449 |
| p030 | 3 | 6570 | 0.6232 | 0.3430 | 0.0338 |
| p060 | 3 | 6471 | 0.0287 | 0.6033 | 0.3680 |

## Safe interpretation

This is the robust-target counterpart of F13. It is intended for diagnostic calibrator training, not direct deployment.
