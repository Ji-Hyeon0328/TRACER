# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_124830_manifest.tsv`
- true_csv: `datasets/phase_f/f3_m060_r2_20260721_124829_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_m060_r2_20260721_124829_true_metric_summary.csv`
- n_rows: `8394`

## Summary

| metric | value |
|---|---:|
| n_rows | 8394 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.475250 |
| base_x_start | 0.592086 |
| base_x_end | 7.969435 |
| base_y_start | -0.599053 |
| base_y_end | -0.793857 |
| base_dx | 7.377349 |
| base_dy | -0.194805 |
| base_vx_mean | 0.171495 |
| base_vx_std | 0.033774 |
| base_vy_mean | -0.004705 |
| base_vy_abs_mean | 0.035598 |
| joint_abs_power_mean | 160.581277 |
| joint_abs_power_max | 1011.189989 |
| joint_abs_effort_mean | 39.404490 |
| joint_sq_effort_mean | 352.998685 |
| imu_ang_vel_norm_mean | 0.481882 |
| imu_ang_vel_norm_max | 4.209100 |
| imu_acc_norm_mean | 10.503146 |
| imu_acc_norm_max | 33.940961 |
| contact_count_mean | 2.083750 |
| contact_force_z_sum_mean | 124.305848 |
| contact_force_z_sum_max | 800.098151 |
| contact_force_norm_sum_mean | 148.087120 |
| contact_force_norm_sum_max | 887.014911 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
