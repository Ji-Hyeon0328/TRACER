# TRACER Phase-F8 D7 True-Metric Beta Calibrator v0

This trains a ridge calibration model from D7 runtime features to Phase-F6 true-metric beta target seeds.

- input csv: `datasets/phase_f/f7_d7_true_metric_calibration_dataset_v0.csv`
- model json: `models/phase_f/f8_d7_true_metric_beta_calibrator_ridge_v0.json`
- rows: `19814`
- features: `43`
- alpha: `1.0`

## Train error

| beta | RMSE | MAE |
|---|---:|---:|
| motion | 0.000719 | 0.000560 |
| stability | 0.003297 | 0.002583 |
| energy | 0.004007 | 0.003141 |

## Per-condition mean prediction

| tag | rows | target beta | predicted beta | abs error |
|---|---:|---|---|---|
| clean | 2226 | (0.3498, 0.3003, 0.3498) | (0.3495, 0.2989, 0.3515) | (0.0003, 0.0014, 0.0017) |
| clean_r1 | 2224 | (0.2516, 0.3673, 0.3811) | (0.2515, 0.3667, 0.3819) | (0.0001, 0.0006, 0.0007) |
| clean_r2 | 2222 | (0.3865, 0.3758, 0.2376) | (0.3877, 0.3811, 0.2312) | (0.0011, 0.0053, 0.0064) |
| m030 | 2225 | (0.0414, 0.3519, 0.6067) | (0.0414, 0.3519, 0.6067) | (0.0000, 0.0000, 0.0000) |
| m030_r1 | 2122 | (0.3986, 0.1630, 0.4383) | (0.3984, 0.1620, 0.4396) | (0.0002, 0.0011, 0.0013) |
| m030_r2 | 2225 | (0.1182, 0.4908, 0.3910) | (0.1183, 0.4912, 0.3905) | (0.0001, 0.0004, 0.0005) |
| p030 | 2226 | (0.5769, 0.3820, 0.0410) | (0.5769, 0.3820, 0.0411) | (0.0000, 0.0000, 0.0000) |
| p030_r1 | 2223 | (0.2945, 0.3787, 0.3268) | (0.2951, 0.3816, 0.3232) | (0.0006, 0.0029, 0.0036) |
| p030_r2 | 2121 | (0.3448, 0.4635, 0.1917) | (0.3435, 0.4577, 0.1988) | (0.0013, 0.0058, 0.0071) |

## Leave-one-tag-out diagnostic

| heldout | rows | target mean | predicted mean | RMSE |
|---|---:|---|---|---|
| clean | 2226 | (0.3498, 0.3003, 0.3498) | (0.3349, 0.2308, 0.4343) | (0.0150, 0.0695, 0.0844) |
| clean_r1 | 2224 | (0.2516, 0.3673, 0.3811) | (0.2213, 0.2172, 0.5615) | (0.0303, 0.1501, 0.1804) |
| clean_r2 | 2222 | (0.3865, 0.3758, 0.2376) | (0.3905, 0.3941, 0.2154) | (0.0039, 0.0183, 0.0222) |
| m030 | 2225 | (0.0414, 0.3519, 0.6067) | (0.0471, 0.0996, 0.8533) | (0.0057, 0.2523, 0.2467) |
| m030_r1 | 2122 | (0.3986, 0.1630, 0.4383) | (0.3786, 0.0735, 0.5480) | (0.0201, 0.0896, 0.1096) |
| m030_r2 | 2225 | (0.1182, 0.4908, 0.3910) | (0.1711, 0.7169, 0.1120) | (0.0528, 0.2261, 0.2789) |
| p030 | 2226 | (0.5769, 0.3820, 0.0410) | (0.5211, 0.1215, 0.3574) | (0.0558, 0.2606, 0.3164) |
| p030_r1 | 2223 | (0.2945, 0.3787, 0.3268) | (0.3016, 0.4116, 0.2868) | (0.0072, 0.0329, 0.0400) |
| p030_r2 | 2121 | (0.3448, 0.4635, 0.1917) | (0.3410, 0.4460, 0.2130) | (0.0038, 0.0175, 0.0213) |

## Safe interpretation

This is a diagnostic calibration model, not a deployable replacement for D7. The dataset has many timestep rows but only three source rollout conditions, so train error can be optimistic. Use this to quantify the gap between existing D7 runtime beta behavior and true-metric beta seeds.
