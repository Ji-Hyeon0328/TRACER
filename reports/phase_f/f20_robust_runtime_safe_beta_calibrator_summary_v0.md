# TRACER Phase-F20 Robust Runtime-Safe Beta Calibrator v0

This trains a diagnostic beta calibrator using runtime-available D7/log features and Phase-F18 robust median true-metric beta targets.

- input: `datasets/phase_f/f19_d7_robust_beta_calibration_dataset_v0.csv`
- model: `models/phase_f/f20_robust_runtime_safe_beta_calibrator_ridge_v0.json`
- rows: `45402`
- features: `36`
- alpha: `1.0`

## Train error

| beta | RMSE | MAE |
|---|---:|---:|
| motion | 0.183284 | 0.143452 |
| stability | 0.141755 | 0.103755 |
| energy | 0.136876 | 0.112902 |

## Per-condition train-fit mean

| tag | rows | target beta | predicted beta | L1 mean error |
|---|---:|---|---|---:|
| clean | 6672 | (0.3807, 0.2387, 0.3807) | (0.4527, 0.2506, 0.2967) | 0.167880 |
| m015 | 5966 | (0.5301, 0.2194, 0.2505) | (0.4326, 0.2765, 0.2909) | 0.195072 |
| m030 | 6572 | (0.3610, 0.1145, 0.5245) | (0.4069, 0.2587, 0.3344) | 0.380106 |
| m060 | 6677 | (0.6204, 0.1594, 0.2202) | (0.4090, 0.2730, 0.3179) | 0.422714 |
| p015 | 6474 | (0.3885, 0.2666, 0.3449) | (0.4095, 0.2909, 0.2995) | 0.090654 |
| p030 | 6570 | (0.6232, 0.3430, 0.0338) | (0.4384, 0.2862, 0.2754) | 0.483118 |
| p060 | 6471 | (0.0287, 0.6033, 0.3680) | (0.3825, 0.3080, 0.3096) | 0.707450 |

## Leave-one-tag-out diagnostic

| heldout | rows | target mean | predicted mean | mean L1 error | RMSE |
|---|---:|---|---|---:|---|
| clean | 6672 | (0.3807, 0.2387, 0.3807) | (0.5148, 0.2503, 0.2348) | 0.304147 | (0.1620, 0.0283, 0.1621) |
| m015 | 5966 | (0.5301, 0.2194, 0.2505) | (0.3886, 0.3145, 0.2969) | 0.293169 | (0.1965, 0.1670, 0.0540) |
| m030 | 6572 | (0.3610, 0.1145, 0.5245) | (0.4202, 0.2934, 0.2864) | 0.482283 | (0.0770, 0.1875, 0.2405) |
| m060 | 6677 | (0.6204, 0.1594, 0.2202) | (0.3439, 0.3044, 0.3516) | 0.557075 | (0.2835, 0.1531, 0.1411) |
| p015 | 6474 | (0.3885, 0.2666, 0.3449) | (0.4153, 0.2979, 0.2868) | 0.131179 | (0.0453, 0.0440, 0.0731) |
| p030 | 6570 | (0.6232, 0.3430, 0.0338) | (0.3856, 0.2717, 0.3427) | 0.621789 | (0.2404, 0.0836, 0.3096) |
| p060 | 6471 | (0.0287, 0.6033, 0.3680) | (0.4934, 0.2166, 0.2901) | 0.960739 | (0.4667, 0.3896, 0.1184) |

## Safe interpretation

This is the robust-target counterpart of F14/F10. It reduces mean-target sensitivity to rollout outliers, but it is still diagnostic and should not be deployed without broader terrain/objective diversity.
