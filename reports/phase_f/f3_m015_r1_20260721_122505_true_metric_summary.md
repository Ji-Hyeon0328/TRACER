# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_122506_manifest.tsv`
- true_csv: `datasets/phase_f/f3_m015_r1_20260721_122505_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_m015_r1_20260721_122505_true_metric_summary.csv`
- n_rows: `8393`

## Summary

| metric | value |
|---|---:|
| n_rows | 8393 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.458500 |
| base_x_start | 0.600888 |
| base_x_end | 8.073813 |
| base_y_start | -0.153423 |
| base_y_end | -0.491237 |
| base_dx | 7.472925 |
| base_dy | -0.337814 |
| base_vx_mean | 0.173938 |
| base_vx_std | 0.041704 |
| base_vy_mean | -0.008052 |
| base_vy_abs_mean | 0.035271 |
| joint_abs_power_mean | 156.435054 |
| joint_abs_power_max | 1085.848470 |
| joint_abs_effort_mean | 39.386714 |
| joint_sq_effort_mean | 353.509404 |
| imu_ang_vel_norm_mean | 0.457534 |
| imu_ang_vel_norm_max | 3.610325 |
| imu_acc_norm_mean | 10.503142 |
| imu_acc_norm_max | 37.906337 |
| contact_count_mean | 2.095556 |
| contact_force_z_sum_mean | 122.231999 |
| contact_force_z_sum_max | 694.788109 |
| contact_force_norm_sum_mean | 146.122301 |
| contact_force_norm_sum_max | 867.872474 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
