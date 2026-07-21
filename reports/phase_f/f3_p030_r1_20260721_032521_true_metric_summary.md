# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_032523_manifest.tsv`
- true_csv: `datasets/phase_f/f3_p030_r1_20260721_032521_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_p030_r1_20260721_032521_true_metric_summary.csv`
- n_rows: `8394`

## Summary

| metric | value |
|---|---:|
| n_rows | 8394 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.465750 |
| base_x_start | 0.542052 |
| base_x_end | 8.008186 |
| base_y_start | 0.368821 |
| base_y_end | 0.495972 |
| base_dx | 7.466134 |
| base_dy | 0.127151 |
| base_vx_mean | 0.173713 |
| base_vx_std | 0.033370 |
| base_vy_mean | 0.002959 |
| base_vy_abs_mean | 0.036158 |
| joint_abs_power_mean | 159.364558 |
| joint_abs_power_max | 1000.268678 |
| joint_abs_effort_mean | 39.420243 |
| joint_sq_effort_mean | 353.903637 |
| imu_ang_vel_norm_mean | 0.452450 |
| imu_ang_vel_norm_max | 3.980235 |
| imu_acc_norm_mean | 10.499919 |
| imu_acc_norm_max | 38.456814 |
| contact_count_mean | 2.092447 |
| contact_force_z_sum_mean | 124.790810 |
| contact_force_z_sum_max | 620.427073 |
| contact_force_norm_sum_mean | 148.839159 |
| contact_force_norm_sum_max | 898.570027 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
