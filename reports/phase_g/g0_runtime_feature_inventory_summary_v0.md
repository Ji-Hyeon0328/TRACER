# TRACER Phase-G0 Runtime Feature Inventory v0

This audits which runtime/log columns are currently available for Phase-G RAM/context feature strengthening.

## Input files

| source | path | rows | columns |
|---|---|---:|---:|
| f4_rollout_true_metrics | `datasets/phase_f/f4_true_metric_training_table_v0.csv` | 21 | 36 |
| f18_robust_rollout_targets | `datasets/phase_f/f18_rollout_with_robust_group_beta_targets_v0.csv` | 21 | 81 |
| f19_d7_robust_timestep | `datasets/phase_f/f19_d7_robust_beta_calibration_dataset_v0.csv` | 45402 | 65 |
| f21_mean_vs_robust | `datasets/phase_f/f21_mean_vs_robust_calibrator_comparison_v0.csv` | 7 | 12 |
| f22b_feature_ablation | `datasets/phase_f/f22b_runtime_feature_basis_ablation_imputed_v0.csv` | 42 | 16 |

## Column category counts

| source | category | column count |
|---|---|---:|
| f18_robust_rollout_targets | contact | 11 |
| f18_robust_rollout_targets | energy | 14 |
| f18_robust_rollout_targets | manifest_source | 6 |
| f18_robust_rollout_targets | motion_score | 6 |
| f18_robust_rollout_targets | position_lateral | 44 |
| f18_robust_rollout_targets | reference | 7 |
| f18_robust_rollout_targets | target_beta | 2 |
| f18_robust_rollout_targets | true_metric | 39 |
| f18_robust_rollout_targets | uncategorized | 8 |
| f19_d7_robust_timestep | contact | 2 |
| f19_d7_robust_timestep | d7_beta | 12 |
| f19_d7_robust_timestep | energy | 12 |
| f19_d7_robust_timestep | manifest_source | 6 |
| f19_d7_robust_timestep | motion_score | 7 |
| f19_d7_robust_timestep | position_lateral | 38 |
| f19_d7_robust_timestep | ram_proxy | 3 |
| f19_d7_robust_timestep | reference | 5 |
| f19_d7_robust_timestep | target_beta | 10 |
| f19_d7_robust_timestep | true_metric | 20 |
| f19_d7_robust_timestep | uncategorized | 5 |
| f21_mean_vs_robust | manifest_source | 1 |
| f21_mean_vs_robust | position_lateral | 1 |
| f21_mean_vs_robust | uncategorized | 10 |
| f22b_feature_ablation | energy | 3 |
| f22b_feature_ablation | position_lateral | 6 |
| f22b_feature_ablation | uncategorized | 10 |
| f4_rollout_true_metrics | contact | 6 |
| f4_rollout_true_metrics | energy | 5 |
| f4_rollout_true_metrics | manifest_source | 4 |
| f4_rollout_true_metrics | position_lateral | 22 |
| f4_rollout_true_metrics | reference | 2 |
| f4_rollout_true_metrics | true_metric | 14 |
| f4_rollout_true_metrics | uncategorized | 3 |

## Representative columns by category


### f18_robust_rollout_targets

- **contact**: `contact_count_mean`, `contact_force_z_sum_mean`, `contact_force_z_sum_max`, `contact_force_norm_sum_mean`, `contact_force_norm_sum_max`, `contact_load_proxy`, `group_contact_force_z_sum_mean_mean`, `group_contact_force_z_sum_mean_median`, `group_contact_force_z_sum_mean_std`, `group_contact_force_z_sum_mean_min`, `group_contact_force_z_sum_mean_max`
- **energy**: `joint_abs_power_mean`, `joint_abs_power_max`, `joint_abs_effort_mean`, `joint_sq_effort_mean`, `energy_proxy`, `group_joint_abs_power_mean_mean`, `group_joint_abs_power_mean_median`, `group_joint_abs_power_mean_std`, `group_joint_abs_power_mean_min`, `group_joint_abs_power_mean_max`, `group_mean_energy_score`, `group_mean_beta_energy_true_seed`
- **manifest_source**: `tag`, `manifest`, `true_csv`, `summary_csv`, `base_tag`, `group_base_tag`
- **motion_score**: `group_mean_motion_score`, `group_mean_stability_score`, `group_mean_energy_score`, `group_robust_median_motion_score`, `group_robust_median_stability_score`, `group_robust_median_energy_score`
- **position_lateral**: `reset_y`, `summary_csv`, `base_dx`, `base_dy`, `base_vx_mean`, `base_vx_std`, `base_vy_abs_mean`, `joint_abs_power_max`, `imu_ang_vel_norm_max`, `imu_acc_norm_max`, `contact_force_z_sum_max`, `contact_force_norm_sum_max`
- **reference**: `base_vx_mean`, `base_vx_std`, `group_base_vx_mean_mean`, `group_base_vx_mean_median`, `group_base_vx_mean_std`, `group_base_vx_mean_min`, `group_base_vx_mean_max`
- **target_beta**: `group_mean_dominant_beta_true_seed`, `group_robust_median_dominant_beta_true_seed`
- **true_metric**: `base_vx_mean`, `base_vx_std`, `base_vy_abs_mean`, `joint_abs_power_mean`, `joint_abs_power_max`, `imu_ang_vel_norm_mean`, `imu_ang_vel_norm_max`, `imu_acc_norm_mean`, `imu_acc_norm_max`, `contact_force_z_sum_mean`, `contact_force_z_sum_max`, `contact_force_norm_sum_mean`
- **uncategorized**: `n_rows`, `t_ros_end`, `t_ros_start`, `group_n_rollouts`, `group_mean_beta_motion_true_seed`, `group_robust_median_beta_motion_true_seed`, `group_mean_vs_robust_beta_l1`, `group_beta_floor`

### f19_d7_robust_timestep

- **contact**: `rollout_true_metric_contact_force_z_sum_mean`, `robust_group_true_metric_group_contact_force_z_sum_mean_median`
- **d7_beta**: `pred_beta_motion`, `pred_beta_stability`, `pred_beta_energy`, `raw_beta_motion`, `raw_beta_stability`, `raw_beta_energy`, `prior_beta_motion`, `prior_beta_stability`, `prior_beta_energy`, `actual_beta_motion`, `actual_beta_stability`, `actual_beta_energy`
- **energy**: `pred_beta_energy`, `raw_beta_energy`, `prior_beta_energy`, `actual_beta_energy`, `err_beta_energy`, `energy_score`, `effort_proxy`, `target_beta_energy_robust_true_metric`, `target_energy_score_robust_true_metric`, `target_beta_energy_mean_true_metric`, `rollout_true_metric_joint_abs_power_mean`, `robust_group_true_metric_group_joint_abs_power_mean_median`
- **manifest_source**: `f19_source_tag`, `f19_base_tag`, `f19_manifest`, `f19_d7_csv`, `f19_true_csv`, `f19_summary_csv`
- **motion_score**: `motion_score`, `stability_score`, `energy_score`, `effort_proxy`, `target_motion_score_robust_true_metric`, `target_stability_score_robust_true_metric`, `target_energy_score_robust_true_metric`
- **position_lateral**: `context`, `x`, `y`, `reset_y`, `y_mean`, `pred_beta_stability`, `pred_beta_energy`, `raw_beta_stability`, `raw_beta_energy`, `prior_beta_stability`, `prior_beta_energy`, `actual_beta_stability`
- **ram_proxy**: `ram_slip_proxy_mean`, `ram_roughness_proxy_mean`, `ram_sigma_mean`
- **reference**: `ref_vx_mean`, `ref_yaw_rate_mean`, `ref_clearance_mean`, `rollout_true_metric_base_vx_mean`, `robust_group_true_metric_group_base_vx_mean_median`
- **target_beta**: `target_beta_motion_robust_true_metric`, `target_beta_stability_robust_true_metric`, `target_beta_energy_robust_true_metric`, `target_beta_dominant_robust_true_metric`, `target_motion_score_robust_true_metric`, `target_stability_score_robust_true_metric`, `target_energy_score_robust_true_metric`, `target_beta_motion_mean_true_metric`, `target_beta_stability_mean_true_metric`, `target_beta_energy_mean_true_metric`
- **true_metric**: `target_beta_motion_robust_true_metric`, `target_beta_stability_robust_true_metric`, `target_beta_energy_robust_true_metric`, `target_beta_dominant_robust_true_metric`, `target_motion_score_robust_true_metric`, `target_stability_score_robust_true_metric`, `target_energy_score_robust_true_metric`, `target_beta_motion_mean_true_metric`, `target_beta_stability_mean_true_metric`, `target_beta_energy_mean_true_metric`, `rollout_true_metric_base_vx_mean`, `rollout_true_metric_base_vy_abs_mean`
- **uncategorized**: `t_wall`, `err_beta_motion`, `out_topic`, `model_json`, `target_mean_vs_robust_beta_l1`

### f21_mean_vs_robust

- **manifest_source**: `base_tag`
- **position_lateral**: `f17_max_rollout_loo_beta_delta_l1`
- **uncategorized**: `mean_beta`, `robust_beta`, `mean_vs_robust_beta_l1`, `f17_mean_rollout_loo_beta_delta_l1`, `f14_train_l1`, `f20_train_l1`, `train_l1_improvement_f14_minus_f20`, `f14_loto_l1`, `f20_loto_l1`, `loto_l1_improvement_f14_minus_f20`

### f22b_feature_ablation

- **energy**: `target_energy`, `pred_energy`, `rmse_energy`
- **position_lateral**: `target_stability`, `target_energy`, `pred_stability`, `pred_energy`, `rmse_stability`, `rmse_energy`
- **uncategorized**: `suite`, `heldout`, `n_features`, `n_train`, `n_test`, `target_motion`, `pred_motion`, `loto_l1`, `rmse_motion`, `kept_features`

### f4_rollout_true_metrics

- **contact**: `contact_count_mean`, `contact_force_z_sum_mean`, `contact_force_z_sum_max`, `contact_force_norm_sum_mean`, `contact_force_norm_sum_max`, `contact_load_proxy`
- **energy**: `joint_abs_power_mean`, `joint_abs_power_max`, `joint_abs_effort_mean`, `joint_sq_effort_mean`, `energy_proxy`
- **manifest_source**: `tag`, `manifest`, `true_csv`, `summary_csv`
- **position_lateral**: `reset_y`, `summary_csv`, `base_dx`, `base_dy`, `base_vx_mean`, `base_vx_std`, `base_vy_abs_mean`, `joint_abs_power_max`, `imu_ang_vel_norm_max`, `imu_acc_norm_max`, `contact_force_z_sum_max`, `contact_force_norm_sum_max`
- **reference**: `base_vx_mean`, `base_vx_std`
- **true_metric**: `base_vx_mean`, `base_vx_std`, `base_vy_abs_mean`, `joint_abs_power_mean`, `joint_abs_power_max`, `imu_ang_vel_norm_mean`, `imu_ang_vel_norm_max`, `imu_acc_norm_mean`, `imu_acc_norm_max`, `contact_force_z_sum_mean`, `contact_force_z_sum_max`, `contact_force_norm_sum_mean`
- **uncategorized**: `n_rows`, `t_ros_end`, `t_ros_start`

## Desired Phase-G feature availability

| feature group | candidate | status | supported by |
|---|---|---|---|
| command_tracking | commanded-vs-real velocity tracking error | available_or_partially_available | f18_robust_rollout_targets, f19_d7_robust_timestep, f21_mean_vs_robust |
| command_tracking | yaw tracking error | available_or_partially_available | f19_d7_robust_timestep |
| command_tracking | reference-vs-odom mismatch | available_or_partially_available | f18_robust_rollout_targets, f19_d7_robust_timestep, f21_mean_vs_robust |
| slip_mismatch | base_vx/ref_vx ratio | available_or_partially_available | f4_rollout_true_metrics, f18_robust_rollout_targets, f19_d7_robust_timestep |
| slip_mismatch | lateral velocity while commanded yaw is small | available_or_partially_available | f4_rollout_true_metrics, f18_robust_rollout_targets, f19_d7_robust_timestep |
| slip_mismatch | distance progress per commanded velocity | missing_or_needs_derivation |  |
| contact_health | contact asymmetry | available_or_partially_available | f4_rollout_true_metrics, f18_robust_rollout_targets, f19_d7_robust_timestep |
| contact_health | contact loss ratio | available_or_partially_available | f4_rollout_true_metrics, f18_robust_rollout_targets, f19_d7_robust_timestep |
| contact_health | contact force variance | available_or_partially_available | f4_rollout_true_metrics, f18_robust_rollout_targets, f19_d7_robust_timestep |
| energy_efficiency | energy per distance | available_or_partially_available | f4_rollout_true_metrics, f18_robust_rollout_targets, f19_d7_robust_timestep, f22b_feature_ablation |
| energy_efficiency | power per achieved velocity | available_or_partially_available | f4_rollout_true_metrics, f18_robust_rollout_targets, f19_d7_robust_timestep |
| temporal_recovery | early/mid/late lateral drift trend | available_or_partially_available | f4_rollout_true_metrics, f18_robust_rollout_targets, f19_d7_robust_timestep |
| temporal_recovery | hold drift after goal | available_or_partially_available | f4_rollout_true_metrics, f18_robust_rollout_targets, f19_d7_robust_timestep |
| temporal_recovery | recovery slope | missing_or_needs_derivation |  |
| terrain_history | terrain segment transition history | missing_or_needs_derivation |  |
| terrain_history | time spent in rough/slope/goal segments | available_or_partially_available | f4_rollout_true_metrics, f18_robust_rollout_targets, f19_d7_robust_timestep, f21_mean_vs_robust, f22b_feature_ablation |

## Phase-G interpretation

Phase-G should prioritize features that are runtime-observable and explain true-metric beta targets without leaking target metrics. If a desired feature is absent, it should be derived from synchronized odom/reference/contact logs or added to the runtime logger in a later subphase.
