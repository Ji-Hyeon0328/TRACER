# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_120343_manifest.tsv`
- true_csv: `datasets/phase_f/f3_p015_20260721_120342_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_p015_20260721_120342_true_metric_summary.csv`
- n_rows: `8393`

## Summary

| metric | value |
|---|---:|
| n_rows | 8393 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.452250 |
| base_x_start | 0.581344 |
| base_x_end | 8.054719 |
| base_y_start | 0.169881 |
| base_y_end | 0.109106 |
| base_dx | 7.473376 |
| base_dy | -0.060776 |
| base_vx_mean | 0.173846 |
| base_vx_std | 0.037947 |
| base_vy_mean | -0.001579 |
| base_vy_abs_mean | 0.037528 |
| joint_abs_power_mean | 162.338025 |
| joint_abs_power_max | 988.071465 |
| joint_abs_effort_mean | 39.513234 |
| joint_sq_effort_mean | 360.098660 |
| imu_ang_vel_norm_mean | 0.439660 |
| imu_ang_vel_norm_max | 3.824439 |
| imu_acc_norm_mean | 10.549436 |
| imu_acc_norm_max | 38.673221 |
| contact_count_mean | 2.073871 |
| contact_force_z_sum_mean | 124.391684 |
| contact_force_z_sum_max | 896.472929 |
| contact_force_norm_sum_mean | 147.317106 |
| contact_force_norm_sum_max | 1079.899892 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
