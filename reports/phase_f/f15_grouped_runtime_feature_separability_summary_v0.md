# TRACER Phase-F15 Grouped Runtime Feature Separability Audit v0

This audits whether runtime-safe D7/log features separate the grouped true-metric beta target conditions: clean, m030, and p030.

- input: `datasets/phase_f/f13_d7_grouped_beta_calibration_dataset_v0.csv`
- output csv: `datasets/phase_f/f15_grouped_runtime_feature_separability_v0.csv`
- rows: `43279`
- groups: `clean, m015, m030, m060, p015, p030, p060`

## Top separable runtime features

| rank | feature | kind | range/std | clean mean | m030 mean | p030 mean |
|---:|---|---|---:|---:|---:|---:|
| 1 | y | runtime_feature | 1.4218 | -0.162201 | -0.269541 | -0.030182 |
| 2 | ref_yaw_rate_mean | runtime_feature | 1.3552 | 0.002689 | 0.004986 | 0.000889 |
| 3 | rollout_mean_abs_y | runtime_feature | 0.8078 | 0.177913 | 0.274877 | 0.099280 |
| 4 | mean_abs_y_context | runtime_feature | 0.8078 | 0.177915 | 0.274876 | 0.099281 |
| 5 | rollout_max_abs_y | runtime_feature | 0.8072 | 0.179213 | 0.276246 | 0.100600 |
| 6 | max_abs_y_context | runtime_feature | 0.8072 | 0.179209 | 0.276240 | 0.100597 |
| 7 | stability_score | runtime_feature | 0.7890 | 0.921773 | 0.882257 | 0.954685 |
| 8 | hold_drift | runtime_feature | 0.3854 | 0.008178 | 0.009058 | 0.006345 |
| 9 | raw_beta_energy | runtime_feature | 0.3826 | 0.231309 | 0.219578 | 0.236633 |
| 10 | pred_beta_energy | runtime_feature | 0.2965 | 0.232535 | 0.222826 | 0.236502 |
| 11 | raw_beta_stability | runtime_feature | 0.2581 | 0.431045 | 0.451817 | 0.413068 |
| 12 | pred_beta_stability | runtime_feature | 0.2156 | 0.437202 | 0.453794 | 0.422940 |
| 13 | energy_score | runtime_feature | 0.2131 | 0.526662 | 0.517344 | 0.524003 |
| 14 | effort_proxy | runtime_feature | 0.2131 | 0.473338 | 0.482656 | 0.475997 |
| 15 | ref_vx_mean | runtime_feature | 0.1844 | 0.186578 | 0.189959 | 0.188760 |

## Target separability

| target | range/std | clean mean | m030 mean | p030 mean |
|---|---:|---:|---:|---:|
| target_beta_stability_grouped_true_metric | 3.9816 | 0.204777 | 0.196735 | 0.308085 |
| target_beta_motion_grouped_true_metric | 3.5306 | 0.397469 | 0.397597 | 0.660085 |
| target_beta_energy_grouped_true_metric | 2.4343 | 0.397754 | 0.405668 | 0.031830 |

## Safe interpretation

If target beta separation is much larger than runtime-feature separation, the runtime-safe calibrator cannot reliably infer grouped true-metric beta targets from the current feature set. In that case, the next step is to add more informative runtime state/context features or collect more diverse terrain/objective rollouts.
