# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_130339_manifest.tsv`
- true_csv: `datasets/phase_f/f3_p060_r2_20260721_130338_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_p060_r2_20260721_130338_true_metric_summary.csv`
- n_rows: `8391`

## Summary

| metric | value |
|---|---:|
| n_rows | 8391 |
| t_ros_start | 0.000000 |
| t_ros_end | 57.791250 |
| base_x_start | 5.588386 |
| base_x_end | 8.017297 |
| base_y_start | -0.049125 |
| base_y_end | 0.062419 |
| base_dx | 2.428911 |
| base_dy | 0.111544 |
| base_vx_mean | 0.056886 |
| base_vx_std | 0.079114 |
| base_vy_mean | -0.001005 |
| base_vy_abs_mean | 0.024021 |
| joint_abs_power_mean | 155.603657 |
| joint_abs_power_max | 921.173753 |
| joint_abs_effort_mean | 48.672590 |
| joint_sq_effort_mean | 458.247668 |
| imu_ang_vel_norm_mean | 0.310795 |
| imu_ang_vel_norm_max | 3.921991 |
| imu_acc_norm_mean | 12.840333 |
| imu_acc_norm_max | 33.752289 |
| contact_count_mean | 2.030271 |
| contact_force_z_sum_mean | 76.637375 |
| contact_force_z_sum_max | 599.751769 |
| contact_force_norm_sum_mean | 89.813002 |
| contact_force_norm_sum_max | 794.364581 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
