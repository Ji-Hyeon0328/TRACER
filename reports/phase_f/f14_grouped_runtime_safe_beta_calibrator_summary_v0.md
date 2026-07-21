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
| motion | 0.179734 | 0.138569 |
| stability | 0.121379 | 0.083236 |
| energy | 0.160559 | 0.140993 |

## Per-condition train-fit mean

| tag | rows | target beta | predicted beta | L1 mean error |
|---|---:|---|---|---:|
| clean | 6672 | (0.3975, 0.2048, 0.3978) | (0.4840, 0.2127, 0.3032) | 0.189055 |
| m015 | 5966 | (0.5308, 0.0342, 0.4351) | (0.4679, 0.2337, 0.2985) | 0.399044 |
| m030 | 6572 | (0.3976, 0.1967, 0.4057) | (0.4919, 0.2282, 0.2799) | 0.251588 |
| m060 | 6677 | (0.6776, 0.2774, 0.0450) | (0.4845, 0.2590, 0.2565) | 0.422982 |
| p015 | 6474 | (0.4765, 0.2502, 0.2733) | (0.4731, 0.2683, 0.2586) | 0.036153 |
| p030 | 6570 | (0.6601, 0.3081, 0.0318) | (0.4658, 0.2588, 0.2755) | 0.487324 |
| p060 | 4348 | (0.0268, 0.5622, 0.4110) | (0.4327, 0.3047, 0.2626) | 0.811907 |

## Leave-one-tag-out diagnostic

| heldout | rows | target mean | predicted mean | mean L1 error | RMSE |
|---|---:|---|---|---:|---|
| clean | 6672 | (0.3975, 0.2048, 0.3978) | (0.5382, 0.2069, 0.2549) | 0.318511 | (0.1468, 0.0440, 0.1438) |
| m015 | 5966 | (0.5308, 0.0342, 0.4351) | (0.4282, 0.3017, 0.2700) | 0.540499 | (0.1739, 0.2860, 0.1753) |
| m030 | 6572 | (0.3976, 0.1967, 0.4057) | (0.5494, 0.2187, 0.2320) | 0.415820 | (0.1809, 0.1015, 0.1777) |
| m060 | 6677 | (0.6776, 0.2774, 0.0450) | (0.4160, 0.2491, 0.3350) | 0.584602 | (0.2640, 0.0497, 0.2919) |
| p015 | 6474 | (0.4765, 0.2502, 0.2733) | (0.4720, 0.2736, 0.2544) | 0.072795 | (0.0227, 0.0444, 0.0289) |
| p030 | 6570 | (0.6601, 0.3081, 0.0318) | (0.4069, 0.2462, 0.3469) | 0.635706 | (0.2576, 0.0879, 0.3185) |
| p060 | 4348 | (0.0268, 0.5622, 0.4110) | (0.5625, 0.2217, 0.2158) | 1.071512 | (0.5360, 0.3410, 0.1969) |

## Safe interpretation

This is the grouped-target counterpart of F10. It should reduce repeat-level label noise, but it is still diagnostic and should not be deployed without broader terrain/objective diversity.
