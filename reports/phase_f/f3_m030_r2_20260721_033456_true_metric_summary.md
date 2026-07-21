# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_033457_manifest.tsv`
- true_csv: `datasets/phase_f/f3_m030_r2_20260721_033456_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_m030_r2_20260721_033456_true_metric_summary.csv`
- n_rows: `8395`

## Summary

| metric | value |
|---|---:|
| n_rows | 8395 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.515000 |
| base_x_start | 0.564670 |
| base_x_end | 8.006629 |
| base_y_start | -0.307669 |
| base_y_end | -0.349616 |
| base_dx | 7.441959 |
| base_dy | -0.041947 |
| base_vx_mean | 0.172932 |
| base_vx_std | 0.027717 |
| base_vy_mean | -0.001011 |
| base_vy_abs_mean | 0.035909 |
| joint_abs_power_mean | 157.388886 |
| joint_abs_power_max | 920.450015 |
| joint_abs_effort_mean | 38.978847 |
| joint_sq_effort_mean | 351.247318 |
| imu_ang_vel_norm_mean | 0.421976 |
| imu_ang_vel_norm_max | 2.964694 |
| imu_acc_norm_mean | 10.510949 |
| imu_acc_norm_max | 35.802664 |
| contact_count_mean | 2.088624 |
| contact_force_z_sum_mean | 123.625107 |
| contact_force_z_sum_max | 576.433541 |
| contact_force_norm_sum_mean | 147.273445 |
| contact_force_norm_sum_max | 964.982893 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
