# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_115418_manifest.tsv`
- true_csv: `datasets/phase_f/f3_m060_20260721_115247_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_m060_20260721_115247_true_metric_summary.csv`
- n_rows: `8390`

## Summary

| metric | value |
|---|---:|
| n_rows | 8390 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.676250 |
| base_x_start | 0.617269 |
| base_x_end | 8.067484 |
| base_y_start | -0.610834 |
| base_y_end | -0.869064 |
| base_dx | 7.450214 |
| base_dy | -0.258230 |
| base_vx_mean | 0.173255 |
| base_vx_std | 0.034130 |
| base_vy_mean | -0.006208 |
| base_vy_abs_mean | 0.038065 |
| joint_abs_power_mean | 160.723980 |
| joint_abs_power_max | 1150.873502 |
| joint_abs_effort_mean | 39.367572 |
| joint_sq_effort_mean | 352.905050 |
| imu_ang_vel_norm_mean | 0.459921 |
| imu_ang_vel_norm_max | 4.529406 |
| imu_acc_norm_mean | 10.506065 |
| imu_acc_norm_max | 39.853473 |
| contact_count_mean | 2.083552 |
| contact_force_z_sum_mean | 123.369066 |
| contact_force_z_sum_max | 554.967268 |
| contact_force_norm_sum_mean | 146.652046 |
| contact_force_norm_sum_max | 804.858249 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
