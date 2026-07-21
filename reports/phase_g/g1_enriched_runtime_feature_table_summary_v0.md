# TRACER Phase-G1 Enriched Runtime Feature Table v0

This aggregates F19 D7 timestep logs into rollout-level enriched RAM/context features for true-metric beta prediction diagnostics.

- input: `datasets/phase_f/f19_d7_robust_beta_calibration_dataset_v0.csv`
- output csv: `datasets/phase_g/g1_enriched_runtime_feature_table_v0.csv`
- rollout rows: `21`
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
| 1 | `rollout_true_metric_imu_ang_vel_norm_mean_min_g1` | `target_beta_motion_robust_true_metric` | 0.581065 | 0.581065 |
| 2 | `rollout_true_metric_imu_ang_vel_norm_mean_mean_g1` | `target_beta_motion_robust_true_metric` | 0.581065 | 0.581065 |
| 3 | `rollout_true_metric_imu_ang_vel_norm_mean_max_g1` | `target_beta_motion_robust_true_metric` | 0.581065 | 0.581065 |
| 4 | `rollout_true_metric_joint_abs_power_mean_min_g1` | `target_beta_energy_robust_true_metric` | -0.551477 | 0.551477 |
| 5 | `rollout_true_metric_joint_abs_power_mean_mean_g1` | `target_beta_energy_robust_true_metric` | -0.551477 | 0.551477 |
| 6 | `rollout_true_metric_joint_abs_power_mean_max_g1` | `target_beta_energy_robust_true_metric` | -0.551477 | 0.551477 |
| 7 | `ram_sigma_mean_min_g1` | `target_beta_motion_robust_true_metric` | 0.542008 | 0.542008 |
| 8 | `ram_roughness_proxy_mean_min_g1` | `target_beta_motion_robust_true_metric` | 0.542008 | 0.542008 |
| 9 | `rollout_true_metric_contact_force_z_sum_mean_min_g1` | `target_beta_stability_robust_true_metric` | -0.510632 | 0.510632 |
| 10 | `rollout_true_metric_contact_force_z_sum_mean_mean_g1` | `target_beta_stability_robust_true_metric` | -0.510632 | 0.510632 |
| 11 | `rollout_true_metric_contact_force_z_sum_mean_max_g1` | `target_beta_stability_robust_true_metric` | -0.510632 | 0.510632 |
| 12 | `rollout_true_metric_contact_force_z_sum_mean_min_g1` | `target_beta_motion_robust_true_metric` | 0.496672 | 0.496672 |
| 13 | `rollout_true_metric_contact_force_z_sum_mean_mean_g1` | `target_beta_motion_robust_true_metric` | 0.496672 | 0.496672 |
| 14 | `rollout_true_metric_contact_force_z_sum_mean_max_g1` | `target_beta_motion_robust_true_metric` | 0.496672 | 0.496672 |
| 15 | `ram_sigma_mean_min_g1` | `target_beta_stability_robust_true_metric` | -0.487898 | 0.487898 |
| 16 | `ram_roughness_proxy_mean_min_g1` | `target_beta_stability_robust_true_metric` | -0.487898 | 0.487898 |
| 17 | `rollout_true_metric_imu_ang_vel_norm_mean_min_g1` | `target_beta_stability_robust_true_metric` | -0.487095 | 0.487095 |
| 18 | `rollout_true_metric_imu_ang_vel_norm_mean_mean_g1` | `target_beta_stability_robust_true_metric` | -0.487095 | 0.487095 |
| 19 | `rollout_true_metric_imu_ang_vel_norm_mean_max_g1` | `target_beta_stability_robust_true_metric` | -0.487095 | 0.487095 |
| 20 | `power_per_vx_g1` | `target_beta_stability_robust_true_metric` | 0.484075 | 0.484075 |
| 21 | `err_beta_energy_std_g1` | `target_beta_motion_robust_true_metric` | 0.477980 | 0.477980 |
| 22 | `rollout_true_metric_base_vx_mean_min_g1` | `target_beta_stability_robust_true_metric` | -0.453867 | 0.453867 |
| 23 | `rollout_true_metric_base_vx_mean_mean_g1` | `target_beta_stability_robust_true_metric` | -0.453867 | 0.453867 |
| 24 | `rollout_true_metric_base_vx_mean_max_g1` | `target_beta_stability_robust_true_metric` | -0.453867 | 0.453867 |
| 25 | `vx_tracking_ratio_g1` | `target_beta_stability_robust_true_metric` | -0.448113 | 0.448113 |
| 26 | `contact_per_vx_g1` | `target_beta_stability_robust_true_metric` | 0.444360 | 0.444360 |
| 27 | `vx_tracking_error_g1` | `target_beta_stability_robust_true_metric` | 0.441291 | 0.441291 |
| 28 | `vx_tracking_abs_error_g1` | `target_beta_stability_robust_true_metric` | 0.441291 | 0.441291 |
| 29 | `power_per_vx_g1` | `target_beta_motion_robust_true_metric` | -0.436316 | 0.436316 |
| 30 | `rollout_true_metric_base_vx_mean_min_g1` | `target_beta_motion_robust_true_metric` | 0.416738 | 0.416738 |

## Safe interpretation

This table is diagnostic. Some features are rollout-level summaries; before deployment they should be converted into online temporal-window features. Target beta columns are labels and must not be used as input features.
