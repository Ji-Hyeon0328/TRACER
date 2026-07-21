# TRACER Phase-F4 True Metric Training Table v0

This table aggregates Phase-F3 true metric rollout summaries into a compact training/evaluation table for Objective Selector and RAM follow-up work.

- input summaries: `9`
- output csv: `datasets/phase_f/f4_true_metric_training_table_v0.csv`

| tag | reset_y | n_rows | base_vx_mean | base_vy_abs_mean | joint_abs_power_mean | imu_ang_vel_norm_mean | contact_force_z_sum_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.0 | 8396 | 0.17524709153149653 | 0.036286069845550835 | 155.06664449516705 | 0.4369674781932821 | 122.88384638146492 |
| clean_r1 |  | 8391 | 0.1736698279077338 | 0.03607501807126134 | 157.71481717366666 | 0.46489190458134894 | 122.9648450602748 |
| clean_r2 |  | 8391 | 0.1744389632575795 | 0.037180634196047384 | 160.2333324834721 | 0.4432689310724821 | 122.96472158196131 |
| m030 | -0.3 | 8392 | 0.17239314171971465 | 0.03890242857642325 | 157.90142204040933 | 0.46068154324946653 | 122.91956126841355 |
| m030_r1 |  | 8393 | 0.17416136578721125 | 0.03971824864435249 | 157.86869454975155 | 0.4516997920732395 | 124.40729975217741 |
| m030_r2 |  | 8395 | 0.17293204508753202 | 0.0359085025547075 | 157.38888602829294 | 0.42197584638629976 | 123.62510732232782 |
| p030 | 0.3 | 8393 | 0.17425653925684514 | 0.037330245986125006 | 163.99645437692655 | 0.47960739748013675 | 123.09504902663942 |
| p030_r1 |  | 8394 | 0.1737129318344725 | 0.03615795017640614 | 159.3645582416352 | 0.4524503244529645 | 124.79080961682467 |
| p030_r2 |  | 8397 | 0.17388946913218356 | 0.03640548580059789 | 161.59181924687533 | 0.44402735882807653 | 123.72482057623257 |

## Safe interpretation

This is not yet a learned IRL Objective Selector dataset. It is the first true-metric rollout table that connects high-level deployment conditions with physical proxies from ROS1/Gazebo: energy, IMU agitation, contact load, velocity, and lateral drift.
