# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_023923_manifest.tsv`
- true_csv: `datasets/phase_f/f3_m030_20260721_023915_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_m030_20260721_023915_true_metric_summary.csv`
- n_rows: `8392`

## Summary

| metric | value |
|---|---:|
| n_rows | 8392 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.430250 |
| base_x_start | 0.546219 |
| base_x_end | 7.956769 |
| base_y_start | -0.275504 |
| base_y_end | -0.913116 |
| base_dx | 7.410550 |
| base_dy | -0.637612 |
| base_vx_mean | 0.172393 |
| base_vx_std | 0.028999 |
| base_vy_mean | -0.014905 |
| base_vy_abs_mean | 0.038902 |
| joint_abs_power_mean | 157.901422 |
| joint_abs_power_max | 1038.348293 |
| joint_abs_effort_mean | 39.285293 |
| joint_sq_effort_mean | 354.825116 |
| imu_ang_vel_norm_mean | 0.460682 |
| imu_ang_vel_norm_max | 4.972122 |
| imu_acc_norm_mean | 10.535780 |
| imu_acc_norm_max | 36.950667 |
| contact_count_mean | 2.096163 |
| contact_force_z_sum_mean | 122.919561 |
| contact_force_z_sum_max | 619.636828 |
| contact_force_norm_sum_mean | 147.114690 |
| contact_force_norm_sum_max | 902.439637 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
