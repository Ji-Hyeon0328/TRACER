# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_013053_manifest.tsv`
- true_csv: `datasets/phase_f/f1_ros1_true_metrics_smoke_20260721_013158.csv`
- out_csv: `datasets/phase_f/f2_true_metric_rollout_summary_v0.csv`
- n_rows: `989`

## Summary

| metric | value |
|---|---:|
| n_rows | 989 |
| t_ros_start | 0.000000 |
| t_ros_end | 11.365250 |
| base_x_start | 0.228713 |
| base_x_end | 1.204611 |
| base_y_start | 0.020487 |
| base_y_end | -0.085801 |
| base_dx | 0.975897 |
| base_dy | -0.106289 |
| base_vx_mean | 0.192268 |
| base_vx_std | 0.028502 |
| base_vy_mean | -0.021197 |
| base_vy_abs_mean | 0.037952 |
| joint_abs_power_mean | 151.314421 |
| joint_abs_power_max | 948.824146 |
| joint_abs_effort_mean | 37.230724 |
| joint_sq_effort_mean | 314.753841 |
| imu_ang_vel_norm_mean | 0.478162 |
| imu_ang_vel_norm_max | 3.169006 |
| imu_acc_norm_mean | 10.364462 |
| imu_acc_norm_max | 34.715866 |
| contact_count_mean | 2.017189 |
| contact_force_z_sum_mean | 125.416417 |
| contact_force_z_sum_max | 474.217088 |
| contact_force_norm_sum_mean | 147.332227 |
| contact_force_norm_sum_max | 716.472474 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
