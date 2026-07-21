# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_122019_manifest.tsv`
- true_csv: `datasets/phase_f/f3_m060_r1_20260721_122018_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_m060_r1_20260721_122018_true_metric_summary.csv`
- n_rows: `8396`

## Summary

| metric | value |
|---|---:|
| n_rows | 8396 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.200750 |
| base_x_start | 0.539582 |
| base_x_end | 8.009129 |
| base_y_start | -0.531815 |
| base_y_end | -1.178326 |
| base_dx | 7.469547 |
| base_dy | -0.646511 |
| base_vx_mean | 0.173527 |
| base_vx_std | 0.032493 |
| base_vy_mean | -0.014877 |
| base_vy_abs_mean | 0.039744 |
| joint_abs_power_mean | 163.445487 |
| joint_abs_power_max | 995.847715 |
| joint_abs_effort_mean | 39.311169 |
| joint_sq_effort_mean | 350.321771 |
| imu_ang_vel_norm_mean | 0.458860 |
| imu_ang_vel_norm_max | 4.098629 |
| imu_acc_norm_mean | 10.516276 |
| imu_acc_norm_max | 39.821061 |
| contact_count_mean | 2.069676 |
| contact_force_z_sum_mean | 124.947468 |
| contact_force_z_sum_max | 640.852721 |
| contact_force_norm_sum_mean | 148.103190 |
| contact_force_norm_sum_max | 808.474362 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
