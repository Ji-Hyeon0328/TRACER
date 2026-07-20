# TRACER Phase-D7.1 Runtime Feature Alignment Report v0

- offline_csv: `datasets/phase_d7/d7_objective_bootstrap_dataset_v1.csv`
- offline_rows: `115`
- runtime_csv_count: `3`
- runtime_rows: `7262`

## Global feature distribution shift

| feature | off_mean | run_mean | mean_delta | off_std | run_std | std_ratio | off_p05 | run_p05 | off_p95 | run_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reset_y | 0.039130 | 0.001779 | -0.037352 | 0.256048 | 0.001584 | 0.006185 | -0.300000 | -0.000388 | 0.300000 | 0.003351 |
| y_mean | -0.123072 | -0.072880 | 0.050192 | 0.206018 | 0.154132 | 0.748149 | -0.518841 | -0.353870 | 0.062243 | 0.166518 |
| mean_abs_y_context | 0.160564 | 0.131245 | -0.029319 | 0.178896 | 0.108789 | 0.608113 | 0.020266 | 0.010458 | 0.532740 | 0.353484 |
| max_abs_y_context | 0.228465 | 0.132395 | -0.096071 | 0.204956 | 0.108841 | 0.531049 | 0.049601 | 0.011858 | 0.686459 | 0.355117 |
| ram_slip_proxy_mean | 0.089994 | 0.085720 | -0.004274 | 0.091954 | 0.094169 | 1.024088 | 0.000000 | 0.000000 | 0.250000 | 0.250000 |
| ram_roughness_proxy_mean | 0.169978 | 0.158166 | -0.011813 | 0.129299 | 0.126423 | 0.977754 | 0.049889 | 0.050000 | 0.400000 | 0.400000 |
| ram_sigma_mean | 0.189979 | 0.179826 | -0.010152 | 0.124560 | 0.125443 | 1.007095 | 0.049889 | 0.050000 | 0.350000 | 0.350000 |
| motion_score | 0.970549 | 0.573216 | -0.397333 | 0.051296 | 0.246083 | 4.797347 | 0.869958 | 0.192269 | 1.000000 | 0.943506 |
| stability_score | 0.855155 | 0.939265 | 0.084110 | 0.101177 | 0.044051 | 0.435381 | 0.629334 | 0.857953 | 0.954278 | 0.994143 |
| energy_score | 0.556982 | 0.559523 | 0.002541 | 0.150300 | 0.142318 | 0.946897 | 0.407892 | 0.414788 | 0.847047 | 0.853125 |
| effort_proxy | 0.443018 | 0.440477 | -0.002541 | 0.150300 | 0.142318 | 0.946897 | 0.152953 | 0.146875 | 0.592108 | 0.585212 |
| hold_drift | 0.036696 | 0.009721 | -0.026975 | 0.079864 | 0.021098 | 0.264170 | 0.000000 | 0.000000 | 0.184000 | 0.068433 |
| rollout_max_abs_y | 0.387261 | 0.132398 | -0.254863 | 0.242714 | 0.108841 | 0.448432 | 0.076000 | 0.011858 | 0.820000 | 0.355117 |
| rollout_mean_abs_y | 0.170953 | 0.131245 | -0.039709 | 0.118313 | 0.108790 | 0.919506 | 0.026410 | 0.010458 | 0.417836 | 0.353484 |

## Largest mean shifts

| rank | feature | abs_mean_delta | off_mean | run_mean | comment |
|---:|---|---:|---:|---:|---|
| 1 | motion_score | 0.397333 | 0.970549 | 0.573216 | objective-score feature shift |
| 2 | rollout_max_abs_y | 0.254863 | 0.387261 | 0.132398 | rollout/window statistic shift |
| 3 | max_abs_y_context | 0.096071 | 0.228465 | 0.132395 | state/context statistic shift |
| 4 | stability_score | 0.084110 | 0.855155 | 0.939265 | objective-score feature shift |
| 5 | y_mean | 0.050192 | -0.123072 | -0.072880 | state/context statistic shift |
| 6 | rollout_mean_abs_y | 0.039709 | 0.170953 | 0.131245 | rollout/window statistic shift |
| 7 | reset_y | 0.037352 | 0.039130 | 0.001779 | state/context statistic shift |
| 8 | mean_abs_y_context | 0.029319 | 0.160564 | 0.131245 | state/context statistic shift |
| 9 | hold_drift | 0.026975 | 0.036696 | 0.009721 | rollout/window statistic shift |
| 10 | ram_roughness_proxy_mean | 0.011813 | 0.169978 | 0.158166 | RAM proxy distribution shift |

## Context-level key feature means

| context | off_n | run_n | off_stability_score | run_stability_score | off_energy_score | run_energy_score | off_hold_drift | run_hold_drift | off_max_abs_y | run_max_abs_y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| downslope | 23 | 1482 | 0.832027 | 0.924120 | 0.514907 | 0.482074 | 0.036696 | 0.000000 | 0.387261 | 0.189703 |
| flat | 23 | 1314 | 0.890510 | 0.979726 | 0.507265 | 0.505080 | 0.036696 | 0.000000 | 0.387261 | 0.050685 |
| goal_flat | 23 | 1936 | 0.829739 | 0.923213 | 0.846517 | 0.768516 | 0.036696 | 0.036464 | 0.387261 | 0.119042 |
| rough | 23 | 1294 | 0.848780 | 0.918040 | 0.411752 | 0.442131 | 0.036696 | 0.000000 | 0.387261 | 0.204903 |
| unknown | 0 | 5 | nan | 0.994977 | nan | 0.900000 | nan | 0.000000 | nan | 0.012558 |
| upslope | 23 | 1231 | 0.874719 | 0.961638 | 0.504470 | 0.504213 | 0.036696 | 0.000000 | 0.387261 | 0.095907 |

## Diagnosis

- This report compares the D7 offline bootstrap training distribution with runtime shadow features.
- Large shifts indicate why the raw ridge Objective Selector can collapse under online/window statistics.
- The calibrated runtime beta should remain shadow-only until the feature definition is aligned.
- Recommended next step: build a runtime-aligned D7.1 dataset using the same online/window feature computation.

