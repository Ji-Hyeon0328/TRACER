# TRACER Phase-F12 Grouped True-Metric Beta Targets v0

This aggregates Phase-F4 rollout-level true metrics by base condition before deriving beta target seeds. This reduces repeat-level target noise from min-max normalization.

- input: `datasets/phase_f/f4_true_metric_training_table_v0.csv`
- grouped output: `datasets/phase_f/f12_grouped_true_metric_beta_targets_v0.csv`
- rollout annotated output: `datasets/phase_f/f12_rollout_with_group_beta_targets_v0.csv`
- groups: `7`
- rollouts: `20`
- beta_floor: `0.05`

## Group beta targets

| base_tag | reset_y | n | vx_mean | lat_abs_mean | power_mean | imu_mean | contact_mean | motion | stability | energy | beta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| clean | 0.0 | 3 | 0.174452 | 0.036514 | 157.671598 | 0.448376 | 122.937804 | 0.997248 | 0.846694 | 1.000000 | (0.3498, 0.2995, 0.3507) |
| m015 | -0.15 | 3 | 0.163477 | 0.047563 | 159.315079 | 0.487106 | 124.270434 | 0.000000 | 0.000000 | 0.586997 | (0.0678, 0.0678, 0.8643) |
| m030 | -0.3 | 3 | 0.173162 | 0.038176 | 157.719668 | 0.444786 | 123.650656 | 0.880054 | 0.746539 | 0.987920 | (0.3364, 0.2881, 0.3754) |
| m060 | -0.6 | 3 | 0.172759 | 0.037802 | 161.583582 | 0.466887 | 124.207460 | 0.843390 | 0.526952 | 0.016928 | (0.5812, 0.3753, 0.0435) |
| p015 | 0.15 | 3 | 0.174482 | 0.036100 | 159.453428 | 0.448408 | 123.017490 | 1.000000 | 0.853899 | 0.552231 | (0.4108, 0.3536, 0.2356) |
| p030 | 0.3 | 3 | 0.173953 | 0.036631 | 161.650944 | 0.458695 | 123.870226 | 0.951909 | 0.668783 | 0.000000 | (0.5658, 0.4059, 0.0282) |
| p060 | 0.6 | 2 | 0.172782 | 0.035244 | 160.390566 | 0.437898 | 122.712000 | 0.845508 | 1.000000 | 0.316730 | (0.3873, 0.4541, 0.1586) |

## Safe interpretation

This is still a bootstrap target table, not a completed IRL Objective Selector. It is more stable than per-rollout F6 because repeated rollouts are aggregated by base condition before target construction.
