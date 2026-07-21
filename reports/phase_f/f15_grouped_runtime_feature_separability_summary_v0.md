# TRACER Phase-F15 Grouped Runtime Feature Separability Audit v0

This audits whether runtime-safe D7/log features separate the grouped true-metric beta target conditions: clean, m030, and p030.

- input: `datasets/phase_f/f13_d7_grouped_beta_calibration_dataset_v0.csv`
- output csv: `datasets/phase_f/f15_grouped_runtime_feature_separability_v0.csv`
- rows: `32859`
- groups: `clean, m015, m030, m060, p015, p030, p060`

## Top separable runtime features

| rank | feature | kind | range/std | clean mean | m030 mean | p030 mean |
|---:|---|---|---:|---:|---:|---:|
| 1 | y | runtime_feature | 2.2567 | -0.162201 | -0.269541 | -0.030182 |
| 2 | ref_yaw_rate_mean | runtime_feature | 1.9545 | 0.002689 | 0.004986 | 0.000889 |
| 3 | rollout_max_abs_y | runtime_feature | 0.9975 | 0.179213 | 0.276246 | 0.100600 |
| 4 | max_abs_y_context | runtime_feature | 0.9975 | 0.179209 | 0.276240 | 0.100597 |
| 5 | mean_abs_y_context | runtime_feature | 0.9972 | 0.177915 | 0.274876 | 0.099281 |
| 6 | rollout_mean_abs_y | runtime_feature | 0.9972 | 0.177913 | 0.274877 | 0.099280 |
| 7 | stability_score | runtime_feature | 0.9389 | 0.921773 | 0.882257 | 0.954685 |
| 8 | hold_drift | runtime_feature | 0.4091 | 0.008178 | 0.009058 | 0.006345 |
| 9 | raw_beta_energy | runtime_feature | 0.3969 | 0.231309 | 0.219578 | 0.236633 |
| 10 | pred_beta_energy | runtime_feature | 0.3009 | 0.232535 | 0.222826 | 0.236502 |
| 11 | raw_beta_stability | runtime_feature | 0.2620 | 0.431045 | 0.451817 | 0.413068 |
| 12 | pred_beta_stability | runtime_feature | 0.2246 | 0.437202 | 0.453794 | 0.422940 |
| 13 | raw_beta_motion | runtime_feature | 0.1829 | 0.337646 | 0.328605 | 0.350298 |
| 14 | pred_beta_motion | runtime_feature | 0.1623 | 0.330263 | 0.323380 | 0.340558 |
| 15 | effort_proxy | runtime_feature | 0.1384 | 0.473338 | 0.482656 | 0.475997 |

## Target separability

| target | range/std | clean mean | m030 mean | p030 mean |
|---|---:|---:|---:|---:|
| target_beta_stability_grouped_true_metric | 4.4437 | 0.216564 | 0.165143 | 0.254222 |
| target_beta_motion_grouped_true_metric | 3.9504 | 0.390984 | 0.348441 | 0.615229 |
| target_beta_energy_grouped_true_metric | 2.8759 | 0.392452 | 0.486417 | 0.130550 |

## Safe interpretation

If target beta separation is much larger than runtime-feature separation, the runtime-safe calibrator cannot reliably infer grouped true-metric beta targets from the current feature set. In that case, the next step is to add more informative runtime state/context features or collect more diverse terrain/objective rollouts.
