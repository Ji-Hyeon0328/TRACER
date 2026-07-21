# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_120820_manifest.tsv`
- true_csv: `datasets/phase_f/f3_p060_20260721_120819_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_p060_20260721_120819_true_metric_summary.csv`
- n_rows: `8393`

## Summary

| metric | value |
|---|---:|
| n_rows | 8393 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.453250 |
| base_x_start | 0.605607 |
| base_x_end | 7.922792 |
| base_y_start | 0.622489 |
| base_y_end | 1.022476 |
| base_dx | 7.317185 |
| base_dy | 0.399987 |
| base_vx_mean | 0.170188 |
| base_vx_std | 0.029583 |
| base_vy_mean | 0.009323 |
| base_vy_abs_mean | 0.035337 |
| joint_abs_power_mean | 159.302446 |
| joint_abs_power_max | 998.698059 |
| joint_abs_effort_mean | 39.405971 |
| joint_sq_effort_mean | 359.053378 |
| imu_ang_vel_norm_mean | 0.421534 |
| imu_ang_vel_norm_max | 3.989939 |
| imu_acc_norm_mean | 10.564183 |
| imu_acc_norm_max | 37.701276 |
| contact_count_mean | 2.099726 |
| contact_force_z_sum_mean | 122.435382 |
| contact_force_z_sum_max | 552.089593 |
| contact_force_norm_sum_mean | 146.155183 |
| contact_force_norm_sum_max | 876.374569 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
