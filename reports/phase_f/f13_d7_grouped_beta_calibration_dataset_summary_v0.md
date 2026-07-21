# TRACER Phase-F13 D7 Grouped-Beta Calibration Dataset v0

This attaches Phase-F12 grouped true-metric beta targets to D7 runtime shadow logs from Phase-F3/F11 rollouts.

- input F12 rollout table: `datasets/phase_f/f12_rollout_with_group_beta_targets_v0.csv`
- output csv: `datasets/phase_f/f13_d7_grouped_beta_calibration_dataset_v0.csv`
- total rows: `43279`
- missing sources: `0`

## Rows by base condition

| base_tag | rollouts | output_rows | beta_motion | beta_stability | beta_energy |
|---|---:|---:|---:|---:|---:|
| clean | 3 | 6672 | 0.3498 | 0.2995 | 0.3507 |
| m015 | 3 | 5966 | 0.0678 | 0.0678 | 0.8643 |
| m030 | 3 | 6572 | 0.3364 | 0.2881 | 0.3754 |
| m060 | 3 | 6677 | 0.5812 | 0.3753 | 0.0435 |
| p015 | 3 | 6474 | 0.4108 | 0.3536 | 0.2356 |
| p030 | 3 | 6570 | 0.5658 | 0.4059 | 0.0282 |
| p060 | 2 | 4348 | 0.3873 | 0.4541 | 0.1586 |

## Safe interpretation

This is the grouped-target version of F7. It is better conditioned than per-rollout F7 because repeated rollouts under the same base condition share the same grouped true-metric beta target.
