# TRACER Phase-D7.1b Runtime-Aligned Objective Dataset Summary v1

- manifest: `reports/phase_d4_context_meta_repeat_20260719_232254_manifest.tsv`
- out_csv: `datasets/phase_d7/d7_runtime_aligned_objective_dataset_v1.csv`
- samples: `2419`
- stride: `3`
- max_per_context: `700`
- lateral_bound: `2.0`

## Context counts and beta means

| context | n | beta_m | beta_s | beta_e | motion_score | stability_score | energy_score | rollout_max_abs_y | hold_drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| downslope | 494 | 0.282 | 0.543 | 0.175 | 0.669 | 0.924 | 0.482 | 0.190 | 0.000 |
| flat | 439 | 0.525 | 0.203 | 0.272 | 0.225 | 0.980 | 0.505 | 0.051 | 0.000 |
| goal_flat | 645 | 0.193 | 0.534 | 0.273 | 0.895 | 0.923 | 0.768 | 0.119 | 0.036 |
| rough | 431 | 0.306 | 0.486 | 0.208 | 0.524 | 0.918 | 0.442 | 0.205 | 0.000 |
| upslope | 410 | 0.422 | 0.337 | 0.241 | 0.376 | 0.962 | 0.504 | 0.096 | 0.000 |

## Notes

- This dataset uses the same online/window feature definitions produced by the D7 runtime shadow node.
- Targets are runtime-aligned bootstrap beta labels, not preference-based IRL labels yet.
- This is intended to train the first runtime-aligned Objective Selector model.

