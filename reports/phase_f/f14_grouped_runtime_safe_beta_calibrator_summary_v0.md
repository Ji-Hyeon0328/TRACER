# TRACER Phase-F14 Grouped Runtime-Safe Beta Calibrator v0

This trains a diagnostic beta calibrator using runtime-available D7/log features and Phase-F12 grouped true-metric beta targets.

- input: `datasets/phase_f/f13_d7_grouped_beta_calibration_dataset_v0.csv`
- model: `models/phase_f/f14_grouped_runtime_safe_beta_calibrator_ridge_v0.json`
- rows: `45402`
- features: `36`
- alpha: `1.0`

## Train error

| beta | RMSE | MAE |
|---|---:|---:|
| motion | 0.196826 | 0.151203 |
| stability | 0.135002 | 0.097664 |
| energy | 0.158759 | 0.138266 |

## Per-condition train-fit mean

| tag | rows | target beta | predicted beta | L1 mean error |
|---|---:|---|---|---:|
| clean | 6672 | (0.3975, 0.2048, 0.3978) | (0.4797, 0.2158, 0.3045) | 0.186532 |
| m015 | 5966 | (0.5308, 0.0342, 0.4351) | (0.4606, 0.2389, 0.3005) | 0.409489 |
| m030 | 6572 | (0.3976, 0.1967, 0.4057) | (0.4425, 0.2649, 0.2926) | 0.226110 |
| m060 | 6677 | (0.6776, 0.2774, 0.0450) | (0.4484, 0.2853, 0.2663) | 0.458388 |
| p015 | 6474 | (0.4765, 0.2502, 0.2733) | (0.4481, 0.2865, 0.2653) | 0.072647 |
| p030 | 6570 | (0.6601, 0.3081, 0.0318) | (0.4707, 0.2546, 0.2747) | 0.485782 |
| p060 | 6471 | (0.0268, 0.5622, 0.4110) | (0.4180, 0.3029, 0.2791) | 0.782532 |

## Leave-one-tag-out diagnostic

| heldout | rows | target mean | predicted mean | mean L1 error | RMSE |
|---|---:|---|---|---:|---|
| clean | 6672 | (0.3975, 0.2048, 0.3978) | (0.5446, 0.2017, 0.2536) | 0.356959 | (0.1693, 0.0711, 0.1459) |
| m015 | 5966 | (0.5308, 0.0342, 0.4351) | (0.4225, 0.3058, 0.2717) | 0.554850 | (0.1822, 0.2920, 0.1735) |
| m030 | 6572 | (0.3976, 0.1967, 0.4057) | (0.4552, 0.2799, 0.2650) | 0.286960 | (0.0755, 0.0997, 0.1487) |
| m060 | 6677 | (0.6776, 0.2774, 0.0450) | (0.3776, 0.2850, 0.3374) | 0.625265 | (0.3068, 0.0534, 0.2944) |
| p015 | 6474 | (0.4765, 0.2502, 0.2733) | (0.4418, 0.2957, 0.2626) | 0.116401 | (0.0503, 0.0547, 0.0309) |
| p030 | 6570 | (0.6601, 0.3081, 0.0318) | (0.4164, 0.2394, 0.3442) | 0.626425 | (0.2464, 0.0867, 0.3161) |
| p060 | 6471 | (0.0268, 0.5622, 0.4110) | (0.5403, 0.2211, 0.2386) | 1.027067 | (0.5150, 0.3415, 0.1780) |

## Safe interpretation

This is the grouped-target counterpart of F10. It should reduce repeat-level label noise, but it is still diagnostic and should not be deployed without broader terrain/objective diversity.
