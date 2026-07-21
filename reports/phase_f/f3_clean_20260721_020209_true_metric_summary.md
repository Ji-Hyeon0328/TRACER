# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_020213_manifest.tsv`
- true_csv: `datasets/phase_f/f3_clean_20260721_020209_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_clean_20260721_020209_true_metric_summary.csv`
- n_rows: `8396`

## Summary

| metric | value |
|---|---:|
| n_rows | 8396 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.472000 |
| base_x_start | 0.519925 |
| base_x_end | 8.051743 |
| base_y_start | 0.019919 |
| base_y_end | -0.255860 |
| base_dx | 7.531817 |
| base_dy | -0.275779 |
| base_vx_mean | 0.175247 |
| base_vx_std | 0.028923 |
| base_vy_mean | -0.006602 |
| base_vy_abs_mean | 0.036286 |
| joint_abs_power_mean | 155.066644 |
| joint_abs_power_max | 944.677511 |
| joint_abs_effort_mean | 39.337847 |
| joint_sq_effort_mean | 352.956247 |
| imu_ang_vel_norm_mean | 0.436967 |
| imu_ang_vel_norm_max | 3.370629 |
| imu_acc_norm_mean | 10.500791 |
| imu_acc_norm_max | 35.840025 |
| contact_count_mean | 2.103502 |
| contact_force_z_sum_mean | 122.883846 |
| contact_force_z_sum_max | 658.098600 |
| contact_force_norm_sum_mean | 147.392424 |
| contact_force_norm_sum_max | 953.361931 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
