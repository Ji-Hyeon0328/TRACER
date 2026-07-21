# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_115906_manifest.tsv`
- true_csv: `datasets/phase_f/f3_m015_20260721_115905_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_m015_20260721_115905_true_metric_summary.csv`
- n_rows: `8393`

## Summary

| metric | value |
|---|---:|
| n_rows | 8393 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.407250 |
| base_x_start | 0.575962 |
| base_x_end | 8.104834 |
| base_y_start | -0.118734 |
| base_y_end | -0.193187 |
| base_dx | 7.528872 |
| base_dy | -0.074453 |
| base_vx_mean | 0.175000 |
| base_vx_std | 0.040308 |
| base_vy_mean | -0.001799 |
| base_vy_abs_mean | 0.036111 |
| joint_abs_power_mean | 161.432183 |
| joint_abs_power_max | 894.498057 |
| joint_abs_effort_mean | 39.378962 |
| joint_sq_effort_mean | 349.194102 |
| imu_ang_vel_norm_mean | 0.459819 |
| imu_ang_vel_norm_max | 4.029020 |
| imu_acc_norm_mean | 10.459303 |
| imu_acc_norm_max | 38.935993 |
| contact_count_mean | 2.073394 |
| contact_force_z_sum_mean | 125.401522 |
| contact_force_z_sum_max | 827.630549 |
| contact_force_norm_sum_mean | 148.806713 |
| contact_force_norm_sum_max | 1083.015079 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
