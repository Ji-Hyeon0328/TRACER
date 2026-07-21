# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_032045_manifest.tsv`
- true_csv: `datasets/phase_f/f3_m030_r1_20260721_032044_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_m030_r1_20260721_032044_true_metric_summary.csv`
- n_rows: `8393`

## Summary

| metric | value |
|---|---:|
| n_rows | 8393 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.452500 |
| base_x_start | 0.574585 |
| base_x_end | 8.064908 |
| base_y_start | -0.243608 |
| base_y_end | -0.906310 |
| base_dx | 7.490323 |
| base_dy | -0.662701 |
| base_vx_mean | 0.174161 |
| base_vx_std | 0.036582 |
| base_vy_mean | -0.015510 |
| base_vy_abs_mean | 0.039718 |
| joint_abs_power_mean | 157.868695 |
| joint_abs_power_max | 976.754816 |
| joint_abs_effort_mean | 39.305352 |
| joint_sq_effort_mean | 354.364345 |
| imu_ang_vel_norm_mean | 0.451700 |
| imu_ang_vel_norm_max | 4.465133 |
| imu_acc_norm_mean | 10.511573 |
| imu_acc_norm_max | 36.781061 |
| contact_count_mean | 2.081377 |
| contact_force_z_sum_mean | 124.407300 |
| contact_force_z_sum_max | 704.953216 |
| contact_force_norm_sum_mean | 148.330358 |
| contact_force_norm_sum_max | 935.467802 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
