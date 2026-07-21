# TRACER Phase-F12 Grouped True-Metric Beta Targets v0

This aggregates Phase-F4 rollout-level true metrics by base condition before deriving beta target seeds. This reduces repeat-level target noise from min-max normalization.

- input: `datasets/phase_f/f4_true_metric_training_table_v0.csv`
- grouped output: `datasets/phase_f/f12_grouped_true_metric_beta_targets_v0.csv`
- rollout annotated output: `datasets/phase_f/f12_rollout_with_group_beta_targets_v0.csv`
- groups: `7`
- rollouts: `21`
- beta_floor: `0.05`

## Group beta targets

| base_tag | reset_y | n | vx_mean | lat_abs_mean | power_mean | imu_mean | contact_mean | motion | stability | energy | beta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| clean | 0.0 | 3 | 0.174452 | 0.036514 | 157.671598 | 0.448376 | 122.937804 | 0.999249 | 0.490575 | 1.000000 | (0.3975, 0.2048, 0.3978) |
| m015 | -0.15 | 3 | 0.163477 | 0.047563 | 159.315079 | 0.487106 | 124.270434 | 0.727129 | 0.000000 | 0.586997 | (0.5308, 0.0342, 0.4351) |
| m030 | -0.3 | 3 | 0.173162 | 0.038176 | 157.719668 | 0.444786 | 123.650656 | 0.967270 | 0.453355 | 0.987920 | (0.3976, 0.1967, 0.4057) |
| m060 | -0.6 | 3 | 0.172759 | 0.037802 | 161.583582 | 0.466887 | 124.207460 | 0.957266 | 0.362360 | 0.016928 | (0.6776, 0.2774, 0.0450) |
| p015 | 0.15 | 3 | 0.174482 | 0.036100 | 159.453428 | 0.448408 | 123.017490 | 1.000000 | 0.501338 | 0.552231 | (0.4765, 0.2502, 0.2733) |
| p030 | 0.3 | 3 | 0.173953 | 0.036631 | 161.650944 | 0.458695 | 123.870226 | 0.986877 | 0.433948 | 0.000000 | (0.6601, 0.3081, 0.0318) |
| p060 | 0.6 | 3 | 0.134150 | 0.031503 | 158.794930 | 0.395530 | 107.353792 | 0.000000 | 1.000000 | 0.717710 | (0.0268, 0.5622, 0.4110) |

## Safe interpretation

This is still a bootstrap target table, not a completed IRL Objective Selector. It is more stable than per-rollout F6 because repeated rollouts are aggregated by base condition before target construction.
