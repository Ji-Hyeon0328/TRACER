# TRACER Phase-D7.0c Objective Selector Training Summary v0

- source_csv: `datasets/phase_d7/d7_runtime_aligned_objective_dataset_v1.csv`
- model_json: `models/phase_d7/d7_objective_selector_ridge_runtime_aligned_v1.json`
- model_type: `ridge_regression_beta_selector_v0`
- samples: `2419`
- train/test: `1935` / `484`
- alpha: `0.0001`

## Test metrics

| target | rmse | mae |
|---|---:|---:|
| target_beta_motion | 0.002288 | 0.001786 |
| target_beta_stability | 0.002599 | 0.001931 |
| target_beta_energy | 0.001475 | 0.001041 |

## All-sample metrics

| target | rmse | mae |
|---|---:|---:|
| target_beta_motion | 0.002149 | 0.001626 |
| target_beta_stability | 0.002384 | 0.001788 |
| target_beta_energy | 0.001530 | 0.001029 |

## Notes

- This is the first data-derived Objective Selector model.
- It predicts beta targets from D7 bootstrap features.
- It is not yet preference-based IRL or RL.
- Next step is D7.0d: evaluate predictions by context/reset-y and prepare a runtime shadow node.

