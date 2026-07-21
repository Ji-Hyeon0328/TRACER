# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_122943_manifest.tsv`
- true_csv: `datasets/phase_f/f3_p015_r1_20260721_122942_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_p015_r1_20260721_122942_true_metric_summary.csv`
- n_rows: `8393`

## Summary

| metric | value |
|---|---:|
| n_rows | 8393 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.535000 |
| base_x_start | 0.544156 |
| base_x_end | 8.080087 |
| base_y_start | 0.147883 |
| base_y_end | -0.031510 |
| base_dx | 7.535931 |
| base_dy | -0.179393 |
| base_vx_mean | 0.175349 |
| base_vx_std | 0.045520 |
| base_vy_mean | -0.004409 |
| base_vy_abs_mean | 0.034215 |
| joint_abs_power_mean | 158.323497 |
| joint_abs_power_max | 1063.023134 |
| joint_abs_effort_mean | 39.599746 |
| joint_sq_effort_mean | 363.702467 |
| imu_ang_vel_norm_mean | 0.474771 |
| imu_ang_vel_norm_max | 6.931395 |
| imu_acc_norm_mean | 10.546312 |
| imu_acc_norm_max | 62.609250 |
| contact_count_mean | 2.100083 |
| contact_force_z_sum_mean | 122.185269 |
| contact_force_z_sum_max | 966.954794 |
| contact_force_norm_sum_mean | 146.540807 |
| contact_force_norm_sum_max | 1110.926312 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
