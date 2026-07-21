# TRACER Phase-F18 Robust Grouped True-Metric Beta Targets v0

This builds robust grouped beta targets using group median metrics instead of mean metrics, reducing sensitivity to single-run outliers.

- input: `datasets/phase_f/f4_true_metric_training_table_v0.csv`
- grouped output: `datasets/phase_f/f18_robust_grouped_true_metric_beta_targets_v0.csv`
- rollout annotated output: `datasets/phase_f/f18_rollout_with_robust_group_beta_targets_v0.csv`
- groups: `7`
- rollouts: `21`
- beta_floor: `0.05`

## Robust beta targets

| base_tag | reset_y | n | mean beta | robust median beta | L1 shift | robust dominant |
|---|---:|---:|---|---|---:|---|
| m015 | -0.15 | 3 | (0.5308, 0.0342, 0.4351) | (0.5301, 0.2194, 0.2505) | 0.370508 | motion |
| m060 | -0.6 | 3 | (0.6776, 0.2774, 0.0450) | (0.6204, 0.1594, 0.2202) | 0.350431 | motion |
| m030 | -0.3 | 3 | (0.3976, 0.1967, 0.4057) | (0.3610, 0.1145, 0.5245) | 0.237566 | energy |
| p015 | 0.15 | 3 | (0.4765, 0.2502, 0.2733) | (0.3885, 0.2666, 0.3449) | 0.175938 | motion |
| p060 | 0.6 | 3 | (0.0268, 0.5622, 0.4110) | (0.0287, 0.6033, 0.3680) | 0.086091 | stability |
| p030 | 0.3 | 3 | (0.6601, 0.3081, 0.0318) | (0.6232, 0.3430, 0.0338) | 0.073835 | motion |
| clean | 0.0 | 3 | (0.3975, 0.2048, 0.3978) | (0.3807, 0.2387, 0.3807) | 0.067795 | motion |

## Median metrics used for robust target

| base_tag | vx_med | lat_abs_med | power_med | imu_med | contact_med | motion | stability | energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.174439 | 0.036286 | 157.714817 | 0.443269 | 122.964722 | 1.000000 | 0.608346 | 1.000000 |
| m015 | 0.173938 | 0.036111 | 160.078001 | 0.459819 | 125.177780 | 0.882065 | 0.335780 | 0.390461 |
| m030 | 0.172932 | 0.038902 | 157.868695 | 0.451700 | 123.625107 | 0.645515 | 0.170590 | 0.960310 |
| m060 | 0.173255 | 0.038065 | 160.723980 | 0.459921 | 124.305848 | 0.721384 | 0.148168 | 0.223843 |
| p015 | 0.174252 | 0.036556 | 158.323497 | 0.439660 | 122.475517 | 0.956091 | 0.640381 | 0.843002 |
| p030 | 0.173889 | 0.036405 | 161.591819 | 0.452450 | 123.724821 | 0.870738 | 0.456776 | 0.000000 |
| p060 | 0.170188 | 0.035150 | 159.302446 | 0.421534 | 122.435382 | 0.000000 | 1.000000 | 0.590501 |

## Safe interpretation

This is not a final IRL Objective Selector. It is a robust bootstrap target used to reduce mean-target sensitivity to abnormal rollouts such as very low-velocity/contact episodes.
