# TRACER Phase-F12 Grouped True-Metric Beta Targets v0

This aggregates Phase-F4 rollout-level true metrics by base condition before deriving beta target seeds. This reduces repeat-level target noise from min-max normalization.

- input: `datasets/phase_f/f4_true_metric_training_table_v0.csv`
- grouped output: `datasets/phase_f/f12_grouped_true_metric_beta_targets_v0.csv`
- rollout annotated output: `datasets/phase_f/f12_rollout_with_group_beta_targets_v0.csv`
- groups: `7`
- rollouts: `15`
- beta_floor: `0.05`

## Group beta targets

| base_tag | reset_y | n | vx_mean | lat_abs_mean | power_mean | imu_mean | contact_mean | motion | stability | energy | beta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| clean | 0.0 | 3 | 0.174452 | 0.036514 | 157.671598 | 0.448376 | 122.937804 | 0.996073 | 0.529414 | 1.000000 | (0.3910, 0.2166, 0.3925) |
| m015 | -0.15 | 2 | 0.174469 | 0.035691 | 158.933619 | 0.458676 | 123.816760 | 1.000000 | 0.456968 | 0.729553 | (0.4494, 0.2170, 0.3336) |
| m030 | -0.3 | 3 | 0.173162 | 0.038176 | 157.719668 | 0.444786 | 123.650656 | 0.694780 | 0.302987 | 0.989699 | (0.3484, 0.1651, 0.4864) |
| m060 | -0.6 | 2 | 0.173391 | 0.038905 | 162.084734 | 0.459390 | 124.158267 | 0.748152 | 0.017897 | 0.054279 | (0.8226, 0.0700, 0.1075) |
| p015 | 0.15 | 1 | 0.173846 | 0.037528 | 162.338025 | 0.439660 | 124.391684 | 0.854428 | 0.382093 | 0.000000 | (0.6523, 0.3116, 0.0361) |
| p030 | 0.3 | 3 | 0.173953 | 0.036631 | 161.650944 | 0.458695 | 123.870226 | 0.879510 | 0.334088 | 0.147239 | (0.6152, 0.2542, 0.1305) |
| p060 | 0.6 | 1 | 0.170188 | 0.035337 | 159.302446 | 0.421534 | 122.435382 | 0.000000 | 1.000000 | 0.650515 | (0.0278, 0.5832, 0.3891) |

## Safe interpretation

This is still a bootstrap target table, not a completed IRL Objective Selector. It is more stable than per-rollout F6 because repeated rollouts are aggregated by base condition before target construction.
