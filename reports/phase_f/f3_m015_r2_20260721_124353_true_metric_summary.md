# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_124354_manifest.tsv`
- true_csv: `datasets/phase_f/f3_m015_r2_20260721_124353_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_m015_r2_20260721_124353_true_metric_summary.csv`
- n_rows: `8392`

## Summary

| metric | value |
|---|---:|
| n_rows | 8392 |
| t_ros_start | 0.000000 |
| t_ros_end | 47.702250 |
| base_x_start | 0.547586 |
| base_x_end | 6.593201 |
| base_y_start | -0.099860 |
| base_y_end | 1.891175 |
| base_dx | 6.045616 |
| base_dy | 1.991035 |
| base_vx_mean | 0.141493 |
| base_vx_std | 0.070555 |
| base_vy_mean | 0.048311 |
| base_vy_abs_mean | 0.071306 |
| joint_abs_power_mean | 160.078001 |
| joint_abs_power_max | 1160.637175 |
| joint_abs_effort_mean | 40.198899 |
| joint_sq_effort_mean | 359.479858 |
| imu_ang_vel_norm_mean | 0.543966 |
| imu_ang_vel_norm_max | 9.438830 |
| imu_acc_norm_mean | 11.061432 |
| imu_acc_norm_max | 47.506413 |
| contact_count_mean | 2.077216 |
| contact_force_z_sum_mean | 125.177780 |
| contact_force_z_sum_max | 806.049397 |
| contact_force_norm_sum_mean | 148.754858 |
| contact_force_norm_sum_max | 926.682867 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
