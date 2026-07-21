# TRACER Phase-F19 D7 Robust-Beta Calibration Dataset v0

This attaches Phase-F18 robust median true-metric beta targets to D7 runtime shadow logs.

- input F18 rollout table: `datasets/phase_f/f18_rollout_with_robust_group_beta_targets_v0.csv`
- output csv: `datasets/phase_f/f19_d7_robust_beta_calibration_dataset_v0.csv`
- total rows: `47627`
- missing sources: `0`

## Rows by base condition

| base_tag | rollouts | output_rows | beta_motion | beta_stability | beta_energy |
|---|---:|---:|---:|---:|---:|
| clean | 3 | 6672 | 0.3655 | 0.2691 | 0.3655 |
| m015 | 3 | 5966 | 0.4287 | 0.2704 | 0.3009 |
| m030 | 3 | 6572 | 0.3313 | 0.1809 | 0.4878 |
| m060 | 3 | 6677 | 0.4489 | 0.2333 | 0.3178 |
| p015 | 3 | 6474 | 0.3672 | 0.2869 | 0.3460 |
| p030 | 3 | 6570 | 0.4623 | 0.3365 | 0.2012 |
| p045 | 1 | 2225 | 0.5711 | 0.3831 | 0.0458 |
| p060 | 3 | 6471 | 0.0265 | 0.5573 | 0.4162 |

## Safe interpretation

This is the robust-target counterpart of F13. It is intended for diagnostic calibrator training, not direct deployment.
