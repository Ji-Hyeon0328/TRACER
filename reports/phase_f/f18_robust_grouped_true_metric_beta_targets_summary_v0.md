# TRACER Phase-F18 Robust Grouped True-Metric Beta Targets v0

This builds robust grouped beta targets using group median metrics instead of mean metrics, reducing sensitivity to single-run outliers.

- input: `datasets/phase_f/f4_true_metric_training_table_v0.csv`
- grouped output: `datasets/phase_f/f18_robust_grouped_true_metric_beta_targets_v0.csv`
- rollout annotated output: `datasets/phase_f/f18_rollout_with_robust_group_beta_targets_v0.csv`
- groups: `8`
- rollouts: `22`
- beta_floor: `0.05`

## Robust beta targets

| base_tag | reset_y | n | mean beta | robust median beta | L1 shift | robust dominant |
|---|---:|---:|---|---|---:|---|
| m015 | -0.15 | 3 | (0.4707, 0.0589, 0.4704) | (0.4287, 0.2704, 0.3009) | 0.423054 | motion |
| p045 | 0.45 | 1 | (0.7087, 0.2560, 0.0353) | (0.5711, 0.3831, 0.0458) | 0.275049 | motion |
| m060 | -0.6 | 3 | (0.5427, 0.2420, 0.2153) | (0.4489, 0.2333, 0.3178) | 0.204993 | motion |
| m030 | -0.3 | 3 | (0.3931, 0.2043, 0.4026) | (0.3313, 0.1809, 0.4878) | 0.170372 | energy |
| p015 | 0.15 | 3 | (0.4407, 0.2429, 0.3164) | (0.3672, 0.2869, 0.3460) | 0.147177 | motion |
| p030 | 0.3 | 3 | (0.5340, 0.2660, 0.2000) | (0.4623, 0.3365, 0.2012) | 0.143304 | motion |
| clean | 0.0 | 3 | (0.3934, 0.2129, 0.3937) | (0.3655, 0.2691, 0.3655) | 0.112371 | motion |
| p060 | 0.6 | 3 | (0.0255, 0.5348, 0.4397) | (0.0265, 0.5573, 0.4162) | 0.047050 | stability |

## Median metrics used for robust target

| base_tag | vx_med | lat_abs_med | power_med | imu_med | contact_med | motion | stability | energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.174439 | 0.036286 | 157.714817 | 0.443269 | 122.964722 | 1.000000 | 0.723137 | 1.000000 |
| m015 | 0.173938 | 0.036111 | 160.078001 | 0.459819 | 125.177780 | 0.882065 | 0.537978 | 0.604239 |
| m030 | 0.172932 | 0.038902 | 157.868695 | 0.451700 | 123.625107 | 0.645515 | 0.329907 | 0.974230 |
| m060 | 0.173255 | 0.038065 | 160.723980 | 0.459921 | 124.305848 | 0.721384 | 0.350903 | 0.496057 |
| p015 | 0.174252 | 0.036556 | 158.323497 | 0.439660 | 122.475517 | 0.956091 | 0.736112 | 0.898065 |
| p030 | 0.173889 | 0.036405 | 161.591819 | 0.452450 | 123.724821 | 0.870738 | 0.620057 | 0.350720 |
| p045 | 0.172627 | 0.036754 | 163.686050 | 0.499371 | 123.154874 | 0.573670 | 0.368309 | 0.000000 |
| p060 | 0.170188 | 0.035150 | 159.302446 | 0.421534 | 122.435382 | 0.000000 | 1.000000 | 0.734120 |

## Safe interpretation

This is not a final IRL Objective Selector. It is a robust bootstrap target used to reduce mean-target sensitivity to abnormal rollouts such as very low-velocity/contact episodes.
