# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_020740_manifest.tsv`
- true_csv: `datasets/phase_f/f3_p030_20260721_020739_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_p030_20260721_020739_true_metric_summary.csv`
- n_rows: `8393`

## Summary

| metric | value |
|---|---:|
| n_rows | 8393 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.460250 |
| base_x_start | 0.560993 |
| base_x_end | 8.050881 |
| base_y_start | 0.315310 |
| base_y_end | 0.169027 |
| base_dx | 7.489888 |
| base_dy | -0.146283 |
| base_vx_mean | 0.174257 |
| base_vx_std | 0.031495 |
| base_vy_mean | -0.003524 |
| base_vy_abs_mean | 0.037330 |
| joint_abs_power_mean | 163.996454 |
| joint_abs_power_max | 914.882762 |
| joint_abs_effort_mean | 39.967592 |
| joint_sq_effort_mean | 360.616309 |
| imu_ang_vel_norm_mean | 0.479607 |
| imu_ang_vel_norm_max | 3.472612 |
| imu_acc_norm_mean | 10.492141 |
| imu_acc_norm_max | 36.557948 |
| contact_count_mean | 2.091743 |
| contact_force_z_sum_mean | 123.095049 |
| contact_force_z_sum_max | 683.012761 |
| contact_force_norm_sum_mean | 146.883277 |
| contact_force_norm_sum_max | 869.186337 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
