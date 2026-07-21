# TRACER Phase-F2 True Metric Summary v0

This summarizes the Phase-F1 ROS1 true metric logger smoke CSV into rollout-level physical metrics for future Objective Selector / RAM training.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_125823_manifest.tsv`
- true_csv: `datasets/phase_f/f3_p060_r1_20260721_125822_true_metrics.csv`
- out_csv: `datasets/phase_f/f3_p060_r1_20260721_125822_true_metric_summary.csv`
- n_rows: `8392`

## Summary

| metric | value |
|---|---:|
| n_rows | 8392 |
| t_ros_start | 0.000000 |
| t_ros_end | 50.357000 |
| base_x_start | 0.528830 |
| base_x_end | 8.067625 |
| base_y_start | 0.614018 |
| base_y_end | 0.612124 |
| base_dx | 7.538796 |
| base_dy | -0.001894 |
| base_vx_mean | 0.175376 |
| base_vx_std | 0.034891 |
| base_vy_mean | -0.000095 |
| base_vy_abs_mean | 0.035150 |
| joint_abs_power_mean | 161.478686 |
| joint_abs_power_max | 929.386648 |
| joint_abs_effort_mean | 39.695080 |
| joint_sq_effort_mean | 357.360761 |
| imu_ang_vel_norm_mean | 0.454262 |
| imu_ang_vel_norm_max | 5.239916 |
| imu_acc_norm_mean | 10.602040 |
| imu_acc_norm_max | 38.278780 |
| contact_count_mean | 2.087822 |
| contact_force_z_sum_mean | 122.988618 |
| contact_force_z_sum_max | 598.899775 |
| contact_force_norm_sum_mean | 146.591146 |
| contact_force_norm_sum_max | 868.248783 |

## Interpretation

- `joint_abs_power_mean` is the current energy proxy: mean of sum(abs(effort_i * velocity_i)).
- `imu_ang_vel_norm_mean/max` provides an IMU-based stability/agitation proxy.
- `contact_force_z_sum_mean/max` and `contact_count_mean` provide a contact/load proxy.
- `base_vx_mean` and `base_vy_abs_mean` provide motion tracking and lateral drift proxy signals.
