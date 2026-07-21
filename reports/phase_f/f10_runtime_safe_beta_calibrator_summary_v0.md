# TRACER Phase-F10 Runtime-Safe Beta Calibrator v0

This trains a diagnostic beta calibrator using only runtime-available D7/log features, excluding true-metric target leakage.

- input: `datasets/phase_f/f7_d7_true_metric_calibration_dataset_v0.csv`
- model: `models/phase_f/f10_runtime_safe_beta_calibrator_ridge_v0.json`
- rows: `19814`
- features: `36`
- alpha: `1.0`

## Train error

| beta | RMSE | MAE |
|---|---:|---:|
| motion | 0.141659 | 0.114597 |
| stability | 0.070151 | 0.053971 |
| energy | 0.123910 | 0.097565 |

## Per-condition train-fit mean

| tag | rows | target beta | predicted beta | L1 mean error |
|---|---:|---|---|---:|
| clean | 2226 | (0.3498, 0.3003, 0.3498) | (0.2636, 0.3564, 0.3799) | 0.172445 |
| clean_r1 | 2224 | (0.2516, 0.3673, 0.3811) | (0.3501, 0.3518, 0.2981) | 0.197067 |
| clean_r2 | 2222 | (0.3865, 0.3758, 0.2376) | (0.3648, 0.3790, 0.2561) | 0.043415 |
| m030 | 2225 | (0.0414, 0.3519, 0.6067) | (0.2334, 0.3115, 0.4551) | 0.384008 |
| m030_r1 | 2122 | (0.3986, 0.1630, 0.4383) | (0.2799, 0.2856, 0.4345) | 0.245153 |
| m030_r2 | 2225 | (0.1182, 0.4908, 0.3910) | (0.2927, 0.4184, 0.2889) | 0.348922 |
| p030 | 2226 | (0.5769, 0.3820, 0.0410) | (0.3037, 0.4121, 0.2843) | 0.546502 |
| p030_r1 | 2223 | (0.2945, 0.3787, 0.3268) | (0.3290, 0.3903, 0.2807) | 0.092190 |
| p030_r2 | 2121 | (0.3448, 0.4635, 0.1917) | (0.3397, 0.3695, 0.2908) | 0.198158 |

## Leave-one-tag-out diagnostic

| heldout | rows | target mean | predicted mean | mean L1 error | RMSE |
|---|---:|---|---|---:|---|
| clean | 2226 | (0.3498, 0.3003, 0.3498) | (0.2374, 0.3732, 0.3894) | 0.232802 | (0.1145, 0.0785, 0.0631) |
| clean_r1 | 2224 | (0.2516, 0.3673, 0.3811) | (0.4017, 0.3429, 0.2554) | 0.303644 | (0.1522, 0.0291, 0.1305) |
| clean_r2 | 2222 | (0.3865, 0.3758, 0.2376) | (0.3540, 0.3821, 0.2639) | 0.103421 | (0.0393, 0.0334, 0.0555) |
| m030 | 2225 | (0.0414, 0.3519, 0.6067) | (0.4083, 0.2539, 0.3378) | 0.737875 | (0.3864, 0.1346, 0.2787) |
| m030_r1 | 2122 | (0.3986, 0.1630, 0.4383) | (0.1951, 0.3519, 0.4530) | 0.478380 | (0.2158, 0.1904, 0.0967) |
| m030_r2 | 2225 | (0.1182, 0.4908, 0.3910) | (0.3932, 0.3810, 0.2258) | 0.550003 | (0.2785, 0.1121, 0.1768) |
| p030 | 2226 | (0.5769, 0.3820, 0.0410) | (0.1895, 0.4278, 0.3827) | 0.776776 | (0.3876, 0.0577, 0.3430) |
| p030_r1 | 2223 | (0.2945, 0.3787, 0.3268) | (0.3947, 0.3874, 0.2178) | 0.241232 | (0.1319, 0.0377, 0.1311) |
| p030_r2 | 2121 | (0.3448, 0.4635, 0.1917) | (0.3410, 0.3379, 0.3212) | 0.287391 | (0.0340, 0.1346, 0.1530) |

## Safe interpretation

This is stricter than F8 because it removes true-metric leakage features. If leave-one-tag-out is weak, that means the current three-condition dataset is insufficient for a deployable beta calibrator. Use this as a diagnostic before expanding F3 data.
