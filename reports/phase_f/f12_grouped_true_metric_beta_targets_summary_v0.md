# TRACER Phase-F12 Grouped True-Metric Beta Targets v0

This aggregates Phase-F4 rollout-level true metrics by base condition before deriving beta target seeds. This reduces repeat-level target noise from min-max normalization.

- input: `datasets/phase_f/f4_true_metric_training_table_v0.csv`
- grouped output: `datasets/phase_f/f12_grouped_true_metric_beta_targets_v0.csv`
- rollout annotated output: `datasets/phase_f/f12_rollout_with_group_beta_targets_v0.csv`
- groups: `3`
- rollouts: `9`
- beta_floor: `0.05`

## Group beta targets

| base_tag | reset_y | n | vx_mean | lat_abs_mean | power_mean | imu_mean | contact_mean | motion | stability | energy | beta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| clean | 0.0 | 3 | 0.174452 | 0.036514 | 157.671598 | 0.448376 | 122.937804 | 1.000000 | 0.896749 | 1.000000 | (0.3446, 0.3107, 0.3446) |
| m030 | -0.3 | 3 | 0.173162 | 0.038176 | 157.719668 | 0.444786 | 123.650656 | 0.000000 | 0.435323 | 0.987920 | (0.0318, 0.3085, 0.6597) |
| p030 | 0.3 | 3 | 0.173953 | 0.036631 | 161.650944 | 0.458695 | 123.870226 | 0.613126 | 0.418244 | 0.000000 | (0.5613, 0.3964, 0.0423) |

## Safe interpretation

This is still a bootstrap target table, not a completed IRL Objective Selector. It is more stable than per-rollout F6 because repeated rollouts are aggregated by base condition before target construction.
