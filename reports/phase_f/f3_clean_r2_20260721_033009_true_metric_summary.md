# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_033010_manifest.tsv`
- true_csv: `datasets/phase_f/f3_clean_r2_20260721_033009_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_clean_r2_20260721_033009_true_metric_summary.csv`
- n_rows: `8391`

## Summary

| metric | value |
|---|---:|
| n_rows | 8391 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.444000 |
| base_x_start | 0.576815 |
| base_x_end | 8.072903 |
| base_y_start | 0.016580 |
| base_y_end | -0.206074 |
| base_dx | 7.496088 |
| base_dy | -0.222654 |
| base_vx_mean | 0.174439 |
| base_vx_std | 0.029626 |
| base_vy_mean | -0.005146 |
| base_vy_abs_mean | 0.037181 |
| joint_abs_power_mean | 160.233332 |
| joint_abs_power_max | 912.735748 |
| joint_abs_effort_mean | 39.575049 |
| joint_sq_effort_mean | 358.778842 |
| imu_ang_vel_norm_mean | 0.443269 |
| imu_ang_vel_norm_max | 3.920707 |
| imu_acc_norm_mean | 10.508454 |
| imu_acc_norm_max | 36.029648 |
| contact_count_mean | 2.092123 |
| contact_force_z_sum_mean | 122.964722 |
| contact_force_z_sum_max | 702.287387 |
| contact_force_norm_sum_mean | 146.419257 |
| contact_force_norm_sum_max | 1081.406013 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
