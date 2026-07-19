# TRACER Phase-D7.0b Refined Beta Target Summary v1

- input_csv: `datasets/phase_d7/d7_objective_bootstrap_dataset_v0.csv`
- output_csv: `datasets/phase_d7/d7_objective_bootstrap_dataset_v1.csv`
- samples: `115`

## Context-level beta means

| context | n | beta_m_v0 | beta_s_v0 | beta_e_v0 | beta_m_v1 | beta_s_v1 | beta_e_v1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| downslope | 23 | 0.187 | 0.559 | 0.254 | 0.190 | 0.643 | 0.167 |
| flat | 23 | 0.350 | 0.264 | 0.386 | 0.355 | 0.351 | 0.295 |
| goal_flat | 23 | 0.207 | 0.512 | 0.281 | 0.186 | 0.571 | 0.244 |
| rough | 23 | 0.184 | 0.500 | 0.317 | 0.189 | 0.594 | 0.217 |
| upslope | 23 | 0.269 | 0.388 | 0.343 | 0.272 | 0.476 | 0.252 |

## Notes

- v1 reduces global energy over-weighting compared with v0.
- v1 increases stability beta for samples with larger lateral deviation or hold drift.
- v1 is still a bootstrap heuristic target, intended for D7.0c Objective Selector training.

