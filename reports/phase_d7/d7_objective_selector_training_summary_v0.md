# TRACER Phase-D7.0c Objective Selector Training Summary v0

- source_csv: `datasets/phase_d7/d7_objective_bootstrap_dataset_v1.csv`
- model_json: `models/phase_d7/d7_objective_selector_ridge_v0.json`
- model_type: `ridge_regression_beta_selector_v0`
- samples: `115`
- train/test: `92` / `23`
- alpha: `0.0001`

## Test metrics

| target | rmse | mae |
|---|---:|---:|
| target_beta_motion | 0.013640 | 0.007229 |
| target_beta_stability | 0.021445 | 0.012226 |
| target_beta_energy | 0.008006 | 0.005143 |

## All-sample metrics

| target | rmse | mae |
|---|---:|---:|
| target_beta_motion | 0.007808 | 0.004620 |
| target_beta_stability | 0.012743 | 0.007993 |
| target_beta_energy | 0.005564 | 0.003848 |

## Notes

- This is the first data-derived Objective Selector model.
- It predicts beta targets from D7 bootstrap features.
- It is not yet preference-based IRL or RL.
- Next step is D7.0d: evaluate predictions by context/reset-y and prepare a runtime shadow node.

