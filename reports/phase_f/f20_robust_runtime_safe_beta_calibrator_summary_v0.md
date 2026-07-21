# TRACER Phase-F20 Robust Runtime-Safe Beta Calibrator v0

This trains a diagnostic beta calibrator using runtime-available D7/log features and Phase-F18 robust median true-metric beta targets.

- input: `datasets/phase_f/f19_d7_robust_beta_calibration_dataset_v0.csv`
- model: `models/phase_f/f20_robust_runtime_safe_beta_calibrator_ridge_v0.json`
- rows: `47627`
- features: `36`
- alpha: `1.0`

## Train error

| beta | RMSE | MAE |
|---|---:|---:|
| motion | 0.136644 | 0.097946 |
| stability | 0.104717 | 0.076003 |
| energy | 0.098205 | 0.072849 |

## Per-condition train-fit mean

| tag | rows | target beta | predicted beta | L1 mean error |
|---|---:|---|---|---:|
| clean | 6672 | (0.3655, 0.2691, 0.3655) | (0.3899, 0.2829, 0.3272) | 0.076414 |
| m015 | 5966 | (0.4287, 0.2704, 0.3009) | (0.3676, 0.3056, 0.3267) | 0.122076 |
| m030 | 6572 | (0.3313, 0.1809, 0.4878) | (0.3474, 0.2945, 0.3581) | 0.259500 |
| m060 | 6677 | (0.4489, 0.2333, 0.3178) | (0.3448, 0.3070, 0.3482) | 0.208169 |
| p015 | 6474 | (0.3672, 0.2869, 0.3460) | (0.3507, 0.3188, 0.3305) | 0.063938 |
| p030 | 6570 | (0.4623, 0.3365, 0.2012) | (0.3765, 0.3118, 0.3117) | 0.221037 |
| p045 | 2225 | (0.5711, 0.3831, 0.0458) | (0.3581, 0.3236, 0.3184) | 0.545148 |
| p060 | 6471 | (0.0265, 0.5573, 0.4162) | (0.3256, 0.3327, 0.3417) | 0.598082 |

## Leave-one-tag-out diagnostic

| heldout | rows | target mean | predicted mean | mean L1 error | RMSE |
|---|---:|---|---|---:|---|
| clean | 6672 | (0.3655, 0.2691, 0.3655) | (0.4202, 0.2837, 0.2961) | 0.168541 | (0.0849, 0.0278, 0.0830) |
| m015 | 5966 | (0.4287, 0.2704, 0.3009) | (0.3364, 0.3300, 0.3335) | 0.202642 | (0.1509, 0.1157, 0.0428) |
| m030 | 6572 | (0.3313, 0.1809, 0.4878) | (0.3508, 0.3215, 0.3277) | 0.330251 | (0.0361, 0.1461, 0.1631) |
| m060 | 6677 | (0.4489, 0.2333, 0.3178) | (0.3142, 0.3258, 0.3600) | 0.275416 | (0.1429, 0.1000, 0.0573) |
| p015 | 6474 | (0.3672, 0.2869, 0.3460) | (0.3475, 0.3265, 0.3260) | 0.097460 | (0.0329, 0.0449, 0.0381) |
| p030 | 6570 | (0.4623, 0.3365, 0.2012) | (0.3546, 0.3060, 0.3394) | 0.282248 | (0.1109, 0.0461, 0.1394) |
| p045 | 2225 | (0.5711, 0.3831, 0.0458) | (0.3416, 0.3196, 0.3388) | 0.586065 | (0.2299, 0.0668, 0.2936) |
| p060 | 6471 | (0.0265, 0.5573, 0.4162) | (0.4142, 0.2664, 0.3193) | 0.781274 | (0.3886, 0.2929, 0.1148) |

## Safe interpretation

This is the robust-target counterpart of F14/F10. It reduces mean-target sensitivity to rollout outliers, but it is still diagnostic and should not be deployed without broader terrain/objective diversity.
