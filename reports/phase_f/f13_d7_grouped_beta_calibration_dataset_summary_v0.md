# TRACER Phase-F13 D7 Grouped-Beta Calibration Dataset v0

This attaches Phase-F12 grouped true-metric beta targets to D7 runtime shadow logs from Phase-F3/F11 rollouts.

- input F12 rollout table: `datasets/phase_f/f12_rollout_with_group_beta_targets_v0.csv`
- output csv: `datasets/phase_f/f13_d7_grouped_beta_calibration_dataset_v0.csv`
- total rows: `19814`
- missing sources: `0`

## Rows by base condition

| base_tag | rollouts | output_rows | beta_motion | beta_stability | beta_energy |
|---|---:|---:|---:|---:|---:|
| clean | 3 | 6672 | 0.3446 | 0.3107 | 0.3446 |
| m030 | 3 | 6572 | 0.0318 | 0.3085 | 0.6597 |
| p030 | 3 | 6570 | 0.5613 | 0.3964 | 0.0423 |

## Safe interpretation

This is the grouped-target version of F7. It is better conditioned than per-rollout F7 because repeated rollouts under the same base condition share the same grouped true-metric beta target.
