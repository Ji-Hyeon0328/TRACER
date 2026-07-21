# TRACER Phase-F14 Grouped Runtime-Safe Beta Calibrator v0

This trains a diagnostic beta calibrator using runtime-available D7/log features and Phase-F12 grouped true-metric beta targets.

- input: `datasets/phase_f/f13_d7_grouped_beta_calibration_dataset_v0.csv`
- model: `models/phase_f/f14_grouped_runtime_safe_beta_calibrator_ridge_v0.json`
- rows: `43279`
- features: `36`
- alpha: `1.0`

## Train error

| beta | RMSE | MAE |
|---|---:|---:|
| motion | 0.151489 | 0.112352 |
| stability | 0.104991 | 0.076067 |
| energy | 0.248234 | 0.182387 |

## Per-condition train-fit mean

| tag | rows | target beta | predicted beta | L1 mean error |
|---|---:|---|---|---:|
| clean | 6672 | (0.3498, 0.2995, 0.3507) | (0.3618, 0.2921, 0.3460) | 0.024053 |
| m015 | 5966 | (0.0678, 0.0678, 0.8643) | (0.3572, 0.2957, 0.3471) | 1.034445 |
| m030 | 6572 | (0.3364, 0.2881, 0.3754) | (0.3947, 0.3108, 0.2945) | 0.161943 |
| m060 | 6677 | (0.5812, 0.3753, 0.0435) | (0.4164, 0.3307, 0.2530) | 0.418831 |
| p015 | 6474 | (0.4108, 0.3536, 0.2356) | (0.4112, 0.3340, 0.2548) | 0.039308 |
| p030 | 6570 | (0.5658, 0.4059, 0.0282) | (0.3882, 0.3210, 0.2908) | 0.525052 |
| p060 | 4348 | (0.3873, 0.4541, 0.1586) | (0.4045, 0.3446, 0.2510) | 0.219092 |

## Leave-one-tag-out diagnostic

| heldout | rows | target mean | predicted mean | mean L1 error | RMSE |
|---|---:|---|---|---:|---|
| clean | 6672 | (0.3498, 0.2995, 0.3507) | (0.3548, 0.2772, 0.3680) | 0.173297 | (0.0506, 0.0470, 0.0929) |
| m015 | 5966 | (0.0678, 0.0678, 0.8643) | (0.4200, 0.3756, 0.2043) | 1.320555 | (0.3595, 0.3183, 0.6623) |
| m030 | 6572 | (0.3364, 0.2881, 0.3754) | (0.4129, 0.3079, 0.2792) | 0.243888 | (0.0880, 0.0616, 0.1362) |
| m060 | 6677 | (0.5812, 0.3753, 0.0435) | (0.3558, 0.3129, 0.3313) | 0.575520 | (0.2268, 0.0682, 0.2926) |
| p015 | 6474 | (0.4108, 0.3536, 0.2356) | (0.4115, 0.3291, 0.2595) | 0.101122 | (0.0279, 0.0363, 0.0589) |
| p030 | 6570 | (0.5658, 0.4059, 0.0282) | (0.3362, 0.2971, 0.3666) | 0.676819 | (0.2344, 0.1161, 0.3491) |
| p060 | 4348 | (0.3873, 0.4541, 0.1586) | (0.4073, 0.3078, 0.2849) | 0.305315 | (0.0397, 0.1486, 0.1396) |

## Safe interpretation

This is the grouped-target counterpart of F10. It should reduce repeat-level label noise, but it is still diagnostic and should not be deployed without broader terrain/objective diversity.
