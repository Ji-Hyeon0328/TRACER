# TRACER Phase-F13 D7 Grouped-Beta Calibration Dataset v0

This attaches Phase-F12 grouped true-metric beta targets to D7 runtime shadow logs from Phase-F3/F11 rollouts.

- input F12 rollout table: `datasets/phase_f/f12_rollout_with_group_beta_targets_v0.csv`
- output csv: `datasets/phase_f/f13_d7_grouped_beta_calibration_dataset_v0.csv`
- total rows: `43279`
- missing sources: `1`

## Rows by base condition

| base_tag | rollouts | output_rows | beta_motion | beta_stability | beta_energy |
|---|---:|---:|---:|---:|---:|
| clean | 3 | 6672 | 0.3975 | 0.2048 | 0.3978 |
| m015 | 3 | 5966 | 0.5308 | 0.0342 | 0.4351 |
| m030 | 3 | 6572 | 0.3976 | 0.1967 | 0.4057 |
| m060 | 3 | 6677 | 0.6776 | 0.2774 | 0.0450 |
| p015 | 3 | 6474 | 0.4765 | 0.2502 | 0.2733 |
| p030 | 3 | 6570 | 0.6601 | 0.3081 | 0.0318 |
| p060 | 2 | 4348 | 0.0268 | 0.5622 | 0.4110 |

## Safe interpretation

This is the grouped-target version of F7. It is better conditioned than per-rollout F7 because repeated rollouts under the same base condition share the same grouped true-metric beta target.
