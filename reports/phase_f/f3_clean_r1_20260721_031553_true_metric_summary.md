# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_031558_manifest.tsv`
- true_csv: `datasets/phase_f/f3_clean_r1_20260721_031553_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_clean_r1_20260721_031553_true_metric_summary.csv`
- n_rows: `8391`

## Summary

| metric | value |
|---|---:|
| n_rows | 8391 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.429500 |
| base_x_start | 0.550476 |
| base_x_end | 8.016513 |
| base_y_start | 0.078847 |
| base_y_end | -0.300390 |
| base_dx | 7.466037 |
| base_dy | -0.379237 |
| base_vx_mean | 0.173670 |
| base_vx_std | 0.031928 |
| base_vy_mean | -0.009020 |
| base_vy_abs_mean | 0.036075 |
| joint_abs_power_mean | 157.714817 |
| joint_abs_power_max | 955.473091 |
| joint_abs_effort_mean | 39.435894 |
| joint_sq_effort_mean | 354.809878 |
| imu_ang_vel_norm_mean | 0.464892 |
| imu_ang_vel_norm_max | 3.217777 |
| imu_acc_norm_mean | 10.543939 |
| imu_acc_norm_max | 37.573908 |
| contact_count_mean | 2.097843 |
| contact_force_z_sum_mean | 122.964845 |
| contact_force_z_sum_max | 727.237490 |
| contact_force_norm_sum_mean | 146.378363 |
| contact_force_norm_sum_max | 1044.549009 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
