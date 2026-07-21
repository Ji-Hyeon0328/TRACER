# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_125327_manifest.tsv`
- true_csv: `datasets/phase_f/f3_p015_r2_20260721_125326_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_p015_r2_20260721_125326_true_metric_summary.csv`
- n_rows: `8393`

## Summary

| metric | value |
|---|---:|
| n_rows | 8393 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.408250 |
| base_x_start | 0.580294 |
| base_x_end | 8.067018 |
| base_y_start | 0.107103 |
| base_y_end | 0.396030 |
| base_dx | 7.486724 |
| base_dy | 0.288927 |
| base_vx_mean | 0.174252 |
| base_vx_std | 0.030216 |
| base_vy_mean | 0.006570 |
| base_vy_abs_mean | 0.036556 |
| joint_abs_power_mean | 157.698761 |
| joint_abs_power_max | 905.266936 |
| joint_abs_effort_mean | 39.488810 |
| joint_sq_effort_mean | 358.888543 |
| imu_ang_vel_norm_mean | 0.430792 |
| imu_ang_vel_norm_max | 4.316224 |
| imu_acc_norm_mean | 10.545071 |
| imu_acc_norm_max | 37.613065 |
| contact_count_mean | 2.101513 |
| contact_force_z_sum_mean | 122.475517 |
| contact_force_z_sum_max | 813.055668 |
| contact_force_norm_sum_mean | 146.392995 |
| contact_force_norm_sum_max | 903.896038 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
