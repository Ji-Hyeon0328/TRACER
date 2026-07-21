# TRACER Phase-F15 Grouped Runtime Feature Separability Audit v0

This audits whether runtime-safe D7/log features separate the grouped true-metric beta target conditions: clean, m030, and p030.

- input: `datasets/phase_f/f13_d7_grouped_beta_calibration_dataset_v0.csv`
- output csv: `datasets/phase_f/f15_grouped_runtime_feature_separability_v0.csv`
- rows: `19814`
- groups: `clean, m030, p030`

## Top separable runtime features

| rank | feature | kind | range/std | clean mean | m030 mean | p030 mean |
|---:|---|---|---:|---:|---:|---:|
| 1 | y | runtime_feature | 1.1419 | -0.162201 | -0.269541 | -0.030182 |
| 2 | max_abs_y_context | runtime_feature | 0.9567 | 0.179209 | 0.276240 | 0.100597 |
| 3 | rollout_max_abs_y | runtime_feature | 0.9567 | 0.179213 | 0.276246 | 0.100600 |
| 4 | mean_abs_y_context | runtime_feature | 0.9564 | 0.177915 | 0.274876 | 0.099281 |
| 5 | rollout_mean_abs_y | runtime_feature | 0.9564 | 0.177913 | 0.274877 | 0.099280 |
| 6 | ref_yaw_rate_mean | runtime_feature | 0.9273 | 0.002689 | 0.004986 | 0.000889 |
| 7 | stability_score | runtime_feature | 0.8996 | 0.921773 | 0.882257 | 0.954685 |
| 8 | raw_beta_energy | runtime_feature | 0.3984 | 0.231309 | 0.219578 | 0.236633 |
| 9 | pred_beta_energy | runtime_feature | 0.3025 | 0.232535 | 0.222826 | 0.236502 |
| 10 | raw_beta_stability | runtime_feature | 0.2632 | 0.431045 | 0.451817 | 0.413068 |
| 11 | pred_beta_stability | runtime_feature | 0.2187 | 0.437202 | 0.453794 | 0.422940 |
| 12 | raw_beta_motion | runtime_feature | 0.1741 | 0.337646 | 0.328605 | 0.350298 |
| 13 | pred_beta_motion | runtime_feature | 0.1470 | 0.330263 | 0.323380 | 0.340558 |
| 14 | hold_drift | runtime_feature | 0.1264 | 0.008178 | 0.009058 | 0.006345 |
| 15 | energy_score | runtime_feature | 0.0896 | 0.526662 | 0.517344 | 0.524003 |

## Target separability

| target | range/std | clean mean | m030 mean | p030 mean |
|---|---:|---:|---:|---:|
| target_beta_energy_grouped_true_metric | 2.4556 | 0.344630 | 0.659733 | 0.042324 |
| target_beta_motion_grouped_true_metric | 2.4422 | 0.344630 | 0.031781 | 0.561320 |
| target_beta_stability_grouped_true_metric | 2.1514 | 0.310741 | 0.308485 | 0.396357 |

## Safe interpretation

If target beta separation is much larger than runtime-feature separation, the runtime-safe calibrator cannot reliably infer grouped true-metric beta targets from the current feature set. In that case, the next step is to add more informative runtime state/context features or collect more diverse terrain/objective rollouts.
