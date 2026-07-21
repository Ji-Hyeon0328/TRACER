# TRACER Phase-F12 Grouped True-Metric Beta Targets v0

This aggregates Phase-F4 rollout-level true metrics by base condition before deriving beta target seeds. This reduces repeat-level target noise from min-max normalization.

- input: `datasets/phase_f/f4_true_metric_training_table_v0.csv`
- grouped output: `datasets/phase_f/f12_grouped_true_metric_beta_targets_v0.csv`
- rollout annotated output: `datasets/phase_f/f12_rollout_with_group_beta_targets_v0.csv`
- groups: `7`
- rollouts: `13`
- beta_floor: `0.05`

## Group beta targets

| base_tag | reset_y | n | vx_mean | lat_abs_mean | power_mean | imu_mean | contact_mean | motion | stability | energy | beta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| clean | 0.0 | 3 | 0.174452 | 0.036514 | 157.671598 | 0.448376 | 122.937804 | 0.886124 | 0.508379 | 1.000000 | (0.3679, 0.2194, 0.4127) |
| m015 | -0.15 | 1 | 0.175000 | 0.036111 | 161.432183 | 0.459819 | 125.401522 | 1.000000 | 0.328329 | 0.194119 | (0.6278, 0.2262, 0.1460) |
| m030 | -0.3 | 3 | 0.173162 | 0.038176 | 157.719668 | 0.444786 | 123.650656 | 0.618089 | 0.246254 | 0.989699 | (0.3334, 0.1478, 0.5188) |
| m060 | -0.6 | 1 | 0.173255 | 0.038065 | 160.723980 | 0.459921 | 123.369066 | 0.637288 | 0.120494 | 0.345884 | (0.5482, 0.1360, 0.3158) |
| p015 | 0.15 | 1 | 0.173846 | 0.037528 | 162.338025 | 0.439660 | 124.391684 | 0.760115 | 0.364926 | 0.000000 | (0.6354, 0.3254, 0.0392) |
| p030 | 0.3 | 3 | 0.173953 | 0.036631 | 161.650944 | 0.458695 | 123.870226 | 0.782429 | 0.335105 | 0.147239 | (0.5884, 0.2722, 0.1394) |
| p060 | 0.6 | 1 | 0.170188 | 0.035337 | 159.302446 | 0.421534 | 122.435382 | 0.000000 | 1.000000 | 0.650515 | (0.0278, 0.5832, 0.3891) |

## Safe interpretation

This is still a bootstrap target table, not a completed IRL Objective Selector. It is more stable than per-rollout F6 because repeated rollouts are aggregated by base condition before target construction.
