# TRACER Phase-F13 D7 Grouped-Beta Calibration Dataset v0

This attaches Phase-F12 grouped true-metric beta targets to D7 runtime shadow logs from Phase-F3/F11 rollouts.

- input F12 rollout table: `datasets/phase_f/f12_rollout_with_group_beta_targets_v0.csv`
- output csv: `datasets/phase_f/f13_d7_grouped_beta_calibration_dataset_v0.csv`
- total rows: `28513`
- missing sources: `0`

## Rows by base condition

| base_tag | rollouts | output_rows | beta_motion | beta_stability | beta_energy |
|---|---:|---:|---:|---:|---:|
| clean | 3 | 6672 | 0.3679 | 0.2194 | 0.4127 |
| m015 | 1 | 2124 | 0.6278 | 0.2262 | 0.1460 |
| m030 | 3 | 6572 | 0.3334 | 0.1478 | 0.5188 |
| m060 | 1 | 2226 | 0.5482 | 0.1360 | 0.3158 |
| p015 | 1 | 2124 | 0.6354 | 0.3254 | 0.0392 |
| p030 | 3 | 6570 | 0.5884 | 0.2722 | 0.1394 |
| p060 | 1 | 2225 | 0.0278 | 0.5832 | 0.3891 |

## Safe interpretation

This is the grouped-target version of F7. It is better conditioned than per-rollout F7 because repeated rollouts under the same base condition share the same grouped true-metric beta target.
