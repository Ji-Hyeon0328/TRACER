# TRACER Phase-G1 Enriched Runtime Feature Table v0

This aggregates F19 D7 timestep logs into rollout-level enriched RAM/context features for true-metric beta prediction diagnostics.

- input: `datasets/phase_f/f19_d7_robust_beta_calibration_dataset_v0.csv`
- output csv: `datasets/phase_g/g1_enriched_runtime_feature_table_v0.csv`
- rollout rows: `22`
- feature columns: `219`

## Rows by base condition

| base_tag | rollouts |
|---|---:|
| clean | 3 |
| m015 | 3 |
| m030 | 3 |
| m060 | 3 |
| p015 | 3 |
| p030 | 3 |
| p045 | 1 |
| p060 | 3 |

## Key derived features

- `vx_tracking_error_g1`, `vx_tracking_ratio_g1`
- `power_per_vx_g1`, `contact_per_vx_g1`, `imu_per_vx_g1`
- `lateral_per_forward_g1`
- `abs_y_late_minus_early_g1`, `y_late_minus_early_g1`
- beta temporal changes: `*_late_minus_early_g1`
- context fractions: `context_frac_*`

## Top absolute feature-target correlations

| rank | feature | target | corr | abs corr |
|---:|---|---|---:|---:|
| 1 | `rollout_true_metric_imu_ang_vel_norm_mean_min_g1` | `target_beta_motion_robust_true_metric` | 0.629979 | 0.629979 |
| 2 | `rollout_true_metric_imu_ang_vel_norm_mean_mean_g1` | `target_beta_motion_robust_true_metric` | 0.629979 | 0.629979 |
| 3 | `rollout_true_metric_imu_ang_vel_norm_mean_max_g1` | `target_beta_motion_robust_true_metric` | 0.629979 | 0.629979 |
| 4 | `rollout_true_metric_joint_abs_power_mean_min_g1` | `target_beta_energy_robust_true_metric` | -0.582311 | 0.582311 |
| 5 | `rollout_true_metric_joint_abs_power_mean_mean_g1` | `target_beta_energy_robust_true_metric` | -0.582311 | 0.582311 |
| 6 | `rollout_true_metric_joint_abs_power_mean_max_g1` | `target_beta_energy_robust_true_metric` | -0.582311 | 0.582311 |
| 7 | `rollout_true_metric_contact_force_z_sum_mean_min_g1` | `target_beta_motion_robust_true_metric` | 0.530526 | 0.530526 |
| 8 | `rollout_true_metric_contact_force_z_sum_mean_mean_g1` | `target_beta_motion_robust_true_metric` | 0.530526 | 0.530526 |
| 9 | `rollout_true_metric_contact_force_z_sum_mean_max_g1` | `target_beta_motion_robust_true_metric` | 0.530526 | 0.530526 |
| 10 | `ram_sigma_mean_min_g1` | `target_beta_motion_robust_true_metric` | 0.530484 | 0.530484 |
| 11 | `ram_roughness_proxy_mean_min_g1` | `target_beta_motion_robust_true_metric` | 0.530484 | 0.530484 |
| 12 | `rollout_true_metric_contact_force_z_sum_mean_min_g1` | `target_beta_stability_robust_true_metric` | -0.512408 | 0.512408 |
| 13 | `rollout_true_metric_contact_force_z_sum_mean_mean_g1` | `target_beta_stability_robust_true_metric` | -0.512408 | 0.512408 |
| 14 | `rollout_true_metric_contact_force_z_sum_mean_max_g1` | `target_beta_stability_robust_true_metric` | -0.512408 | 0.512408 |
| 15 | `power_per_vx_g1` | `target_beta_stability_robust_true_metric` | 0.491653 | 0.491653 |
| 16 | `power_per_vx_g1` | `target_beta_motion_robust_true_metric` | -0.485768 | 0.485768 |
| 17 | `rollout_true_metric_base_vx_mean_min_g1` | `target_beta_motion_robust_true_metric` | 0.467934 | 0.467934 |
| 18 | `rollout_true_metric_base_vx_mean_mean_g1` | `target_beta_motion_robust_true_metric` | 0.467934 | 0.467934 |
| 19 | `rollout_true_metric_base_vx_mean_max_g1` | `target_beta_motion_robust_true_metric` | 0.467934 | 0.467934 |
| 20 | `vx_tracking_ratio_g1` | `target_beta_motion_robust_true_metric` | 0.464986 | 0.464986 |
| 21 | `ram_sigma_mean_min_g1` | `target_beta_stability_robust_true_metric` | -0.464737 | 0.464737 |
| 22 | `ram_roughness_proxy_mean_min_g1` | `target_beta_stability_robust_true_metric` | -0.464737 | 0.464737 |
| 23 | `rollout_true_metric_base_vx_mean_min_g1` | `target_beta_stability_robust_true_metric` | -0.462825 | 0.462825 |
| 24 | `rollout_true_metric_base_vx_mean_mean_g1` | `target_beta_stability_robust_true_metric` | -0.462825 | 0.462825 |
| 25 | `rollout_true_metric_base_vx_mean_max_g1` | `target_beta_stability_robust_true_metric` | -0.462825 | 0.462825 |
| 26 | `vx_tracking_error_g1` | `target_beta_motion_robust_true_metric` | -0.457572 | 0.457572 |
| 27 | `vx_tracking_abs_error_g1` | `target_beta_motion_robust_true_metric` | -0.457572 | 0.457572 |
| 28 | `contact_per_vx_g1` | `target_beta_motion_robust_true_metric` | -0.456868 | 0.456868 |
| 29 | `vx_tracking_ratio_g1` | `target_beta_stability_robust_true_metric` | -0.456563 | 0.456563 |
| 30 | `contact_per_vx_g1` | `target_beta_stability_robust_true_metric` | 0.452743 | 0.452743 |

## Safe interpretation

This table is diagnostic. Some features are rollout-level summaries; before deployment they should be converted into online temporal-window features. Target beta columns are labels and must not be used as input features.
