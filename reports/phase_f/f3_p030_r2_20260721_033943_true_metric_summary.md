# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_033944_manifest.tsv`
- true_csv: `datasets/phase_f/f3_p030_r2_20260721_033943_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_p030_r2_20260721_033943_true_metric_summary.csv`
- n_rows: `8397`

## Summary

| metric | value |
|---|---:|
| n_rows | 8397 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.549000 |
| base_x_start | 0.600337 |
| base_x_end | 8.080035 |
| base_y_start | 0.277548 |
| base_y_end | 0.296361 |
| base_dx | 7.479698 |
| base_dy | 0.018814 |
| base_vx_mean | 0.173889 |
| base_vx_std | 0.036527 |
| base_vy_mean | 0.000338 |
| base_vy_abs_mean | 0.036405 |
| joint_abs_power_mean | 161.591819 |
| joint_abs_power_max | 992.442709 |
| joint_abs_effort_mean | 39.396734 |
| joint_sq_effort_mean | 352.940082 |
| imu_ang_vel_norm_mean | 0.444027 |
| imu_ang_vel_norm_max | 3.186921 |
| imu_acc_norm_mean | 10.536825 |
| imu_acc_norm_max | 36.427923 |
| contact_count_mean | 2.082887 |
| contact_force_z_sum_mean | 123.724821 |
| contact_force_z_sum_max | 640.419220 |
| contact_force_norm_sum_mean | 146.998553 |
| contact_force_norm_sum_max | 880.758461 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
