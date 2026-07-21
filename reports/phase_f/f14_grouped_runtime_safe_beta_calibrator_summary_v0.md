# TRACER Phase-F14 Grouped Runtime-Safe Beta Calibrator v0

This trains a diagnostic beta calibrator using runtime-available D7/log features and Phase-F12 grouped true-metric beta targets.

- input: `datasets/phase_f/f13_d7_grouped_beta_calibration_dataset_v0.csv`
- model: `models/phase_f/f14_grouped_runtime_safe_beta_calibrator_ridge_v0.json`
- rows: `19814`
- features: `36`
- alpha: `1.0`

## Train error

| beta | RMSE | MAE |
|---|---:|---:|
| motion | 0.178003 | 0.149232 |
| stability | 0.035344 | 0.030192 |
| energy | 0.207615 | 0.173605 |

## Per-condition train-fit mean

| tag | rows | target beta | predicted beta | L1 mean error |
|---|---:|---|---|---:|
| clean | 6672 | (0.3446, 0.3107, 0.3446) | (0.3327, 0.3350, 0.3324) | 0.048439 |
| m030 | 6572 | (0.0318, 0.3085, 0.6597) | (0.2183, 0.3274, 0.4543) | 0.410892 |
| p030 | 6570 | (0.5613, 0.3964, 0.0423) | (0.3866, 0.3527, 0.2607) | 0.436713 |

## Leave-one-tag-out diagnostic

| heldout | rows | target mean | predicted mean | mean L1 error | RMSE |
|---|---:|---|---|---:|---|
| clean | 6672 | (0.3446, 0.3107, 0.3446) | (0.3731, 0.3635, 0.2634) | 0.405341 | (0.1716, 0.0600, 0.2124) |
| m030 | 6572 | (0.0318, 0.3085, 0.6597) | (0.4134, 0.3379, 0.2487) | 0.835139 | (0.3946, 0.0494, 0.4342) |
| p030 | 6570 | (0.5613, 0.3964, 0.0423) | (0.2279, 0.3099, 0.4622) | 0.839701 | (0.3391, 0.0865, 0.4245) |

## Safe interpretation

This is the grouped-target counterpart of F10. It should reduce repeat-level label noise, but it is still diagnostic and should not be deployed without broader terrain/objective diversity.
