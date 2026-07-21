# TRACER Phase-F4 True Metric Training Table v0

This table aggregates Phase-F3 true metric rollout summaries into a compact training/evaluation table for Objective Selector and RAM follow-up work.

- input summaries: `3`
- output csv: `datasets/phase_f/f4_true_metric_training_table_v0.csv`

| tag | reset_y | n_rows | base_vx_mean | base_vy_abs_mean | joint_abs_power_mean | imu_ang_vel_norm_mean | contact_force_z_sum_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.0 | 8396 | 0.17524709153149653 | 0.036286069845550835 | 155.06664449516705 | 0.4369674781932821 | 122.88384638146492 |
| m030 | -0.3 | 8392 | 0.17239314171971465 | 0.03890242857642325 | 157.90142204040933 | 0.46068154324946653 | 122.91956126841355 |
| p030 | 0.3 | 8393 | 0.17425653925684514 | 0.037330245986125006 | 163.99645437692655 | 0.47960739748013675 | 123.09504902663942 |

## Safe interpretation

This is not yet a learned IRL Objective Selector dataset. It is the first true-metric rollout table that connects high-level deployment conditions with physical proxies from ROS1/Gazebo: energy, IMU agitation, contact load, velocity, and lateral drift.
