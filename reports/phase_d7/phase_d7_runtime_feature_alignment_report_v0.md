# TRACER Phase-D7.1 Runtime Feature Alignment Report v0

- offline_csv: `datasets/phase_d7/d7_objective_bootstrap_dataset_v1.csv`
- offline_rows: `115`
- runtime_csv_count: `3`
- runtime_rows: `97652`

## Global feature distribution shift

| feature | off_mean | run_mean | mean_delta | off_std | run_std | std_ratio | off_p05 | run_p05 | off_p95 | run_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reset_y | 0.039130 | 0.005087 | -0.034043 | 0.256048 | 0.005771 | 0.022540 | -0.300000 | -0.001253 | 0.300000 | 0.013141 |
| y_mean | -0.123072 | -0.391687 | -0.268615 | 0.206018 | 0.135217 | 0.656335 | -0.518841 | -0.438648 | 0.062243 | -0.005407 |
| mean_abs_y_context | 0.160564 | 0.397639 | 0.237074 | 0.178896 | 0.113513 | 0.634522 | 0.020266 | 0.047885 | 0.532740 | 0.437723 |
| max_abs_y_context | 0.228465 | 0.398601 | 0.170136 | 0.204956 | 0.113454 | 0.553554 | 0.049601 | 0.049336 | 0.686459 | 0.438648 |
| ram_slip_proxy_mean | 0.089994 | 0.012851 | -0.077143 | 0.091954 | 0.047561 | 0.517222 | 0.000000 | 0.000000 | 0.250000 | 0.100000 |
| ram_roughness_proxy_mean | 0.169978 | 0.066126 | -0.103853 | 0.129299 | 0.061929 | 0.478959 | 0.049889 | 0.050000 | 0.400000 | 0.200000 |
| ram_sigma_mean | 0.189979 | 0.069388 | -0.120591 | 0.124560 | 0.066852 | 0.536710 | 0.049889 | 0.050000 | 0.350000 | 0.300000 |
| motion_score | 0.970549 | 0.837037 | -0.133512 | 0.051296 | 0.117232 | 2.285419 | 0.869958 | 0.558571 | 1.000000 | 0.867857 |
| stability_score | 0.855155 | 0.730024 | -0.125131 | 0.101177 | 0.065669 | 0.649046 | 0.629334 | 0.705750 | 0.954278 | 0.881074 |
| energy_score | 0.556982 | 0.810879 | 0.253897 | 0.150300 | 0.115541 | 0.768734 | 0.407892 | 0.504873 | 0.847047 | 0.853125 |
| effort_proxy | 0.443018 | 0.189121 | -0.253897 | 0.150300 | 0.115541 | 0.768734 | 0.152953 | 0.146875 | 0.592108 | 0.495127 |
| hold_drift | 0.036696 | 0.138167 | 0.101471 | 0.079864 | 0.034669 | 0.434101 | 0.000000 | 0.000000 | 0.184000 | 0.148488 |
| rollout_max_abs_y | 0.387261 | 0.398610 | 0.011349 | 0.242714 | 0.113432 | 0.467349 | 0.076000 | 0.049368 | 0.820000 | 0.438648 |
| rollout_mean_abs_y | 0.170953 | 0.397665 | 0.226712 | 0.118313 | 0.113492 | 0.959253 | 0.026410 | 0.047992 | 0.417836 | 0.437723 |

## Largest mean shifts

| rank | feature | abs_mean_delta | off_mean | run_mean | comment |
|---:|---|---:|---:|---:|---|
| 1 | y_mean | 0.268615 | -0.123072 | -0.391687 | state/context statistic shift |
| 2 | effort_proxy | 0.253897 | 0.443018 | 0.189121 | objective-score feature shift |
| 3 | energy_score | 0.253897 | 0.556982 | 0.810879 | objective-score feature shift |
| 4 | mean_abs_y_context | 0.237074 | 0.160564 | 0.397639 | state/context statistic shift |
| 5 | rollout_mean_abs_y | 0.226712 | 0.170953 | 0.397665 | rollout/window statistic shift |
| 6 | max_abs_y_context | 0.170136 | 0.228465 | 0.398601 | state/context statistic shift |
| 7 | motion_score | 0.133512 | 0.970549 | 0.837037 | objective-score feature shift |
| 8 | stability_score | 0.125131 | 0.855155 | 0.730024 | objective-score feature shift |
| 9 | ram_sigma_mean | 0.120591 | 0.189979 | 0.069388 | RAM proxy distribution shift |
| 10 | ram_roughness_proxy_mean | 0.103853 | 0.169978 | 0.066126 | RAM proxy distribution shift |

## Context-level key feature means

| context | off_n | run_n | off_stability_score | run_stability_score | off_energy_score | run_energy_score | off_hold_drift | run_hold_drift | off_max_abs_y | run_max_abs_y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| downslope | 23 | 2974 | 0.832027 | 0.813436 | 0.514907 | 0.479693 | 0.036696 | 0.074134 | 0.387261 | 0.318075 |
| flat | 23 | 2594 | 0.890510 | 0.928339 | 0.507265 | 0.564254 | 0.036696 | 0.075267 | 0.387261 | 0.028945 |
| goal_flat | 23 | 86971 | 0.829739 | 0.711524 | 0.846517 | 0.849315 | 0.036696 | 0.146006 | 0.387261 | 0.429179 |
| rough | 23 | 2549 | 0.848780 | 0.871598 | 0.411752 | 0.444093 | 0.036696 | 0.073686 | 0.387261 | 0.173589 |
| upslope | 23 | 2564 | 0.874719 | 0.919432 | 0.504470 | 0.505418 | 0.036696 | 0.074297 | 0.387261 | 0.052818 |

## Diagnosis

- This report compares the D7 offline bootstrap training distribution with runtime shadow features.
- Large shifts indicate why the raw ridge Objective Selector can collapse under online/window statistics.
- The calibrated runtime beta should remain shadow-only until the feature definition is aligned.
- Recommended next step: build a runtime-aligned D7.1 dataset using the same online/window feature computation.

