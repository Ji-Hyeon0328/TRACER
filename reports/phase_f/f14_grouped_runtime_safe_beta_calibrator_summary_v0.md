# TRACER Phase-F14 Grouped Runtime-Safe Beta Calibrator v0

This trains a diagnostic beta calibrator using runtime-available D7/log features and Phase-F12 grouped true-metric beta targets.

- input: `datasets/phase_f/f13_d7_grouped_beta_calibration_dataset_v0.csv`
- model: `models/phase_f/f14_grouped_runtime_safe_beta_calibrator_ridge_v0.json`
- rows: `32859`
- features: `36`
- alpha: `1.0`

## Train error

| beta | RMSE | MAE |
|---|---:|---:|
| motion | 0.178044 | 0.147784 |
| stability | 0.086331 | 0.059001 |
| energy | 0.149036 | 0.131258 |

## Per-condition train-fit mean

| tag | rows | target beta | predicted beta | L1 mean error |
|---|---:|---|---|---:|
| clean | 6672 | (0.3910, 0.2166, 0.3925) | (0.4759, 0.2133, 0.3108) | 0.169795 |
| m015 | 4245 | (0.4494, 0.2170, 0.3336) | (0.4972, 0.2096, 0.2932) | 0.095583 |
| m030 | 6572 | (0.3484, 0.1651, 0.4864) | (0.5111, 0.1913, 0.2975) | 0.377810 |
| m060 | 4451 | (0.8226, 0.0700, 0.1075) | (0.5297, 0.1957, 0.2746) | 0.585661 |
| p015 | 2124 | (0.6523, 0.3116, 0.0361) | (0.5355, 0.2017, 0.2628) | 0.453533 |
| p030 | 6570 | (0.6152, 0.2542, 0.1305) | (0.4878, 0.2412, 0.2710) | 0.280830 |
| p060 | 2225 | (0.0278, 0.5832, 0.3891) | (0.2818, 0.4173, 0.3009) | 0.508091 |

## Leave-one-tag-out diagnostic

| heldout | rows | target mean | predicted mean | mean L1 error | RMSE |
|---|---:|---|---|---:|---|
| clean | 6672 | (0.3910, 0.2166, 0.3925) | (0.5168, 0.2156, 0.2675) | 0.285434 | (0.1366, 0.0407, 0.1269) |
| m015 | 4245 | (0.4494, 0.2170, 0.3336) | (0.5067, 0.2085, 0.2848) | 0.132380 | (0.0788, 0.0227, 0.0718) |
| m030 | 6572 | (0.3484, 0.1651, 0.4864) | (0.5877, 0.1964, 0.2160) | 0.554368 | (0.2462, 0.0523, 0.2769) |
| m060 | 4451 | (0.8226, 0.0700, 0.1075) | (0.4256, 0.2364, 0.3380) | 0.793965 | (0.3992, 0.1701, 0.2372) |
| p015 | 2124 | (0.6523, 0.3116, 0.0361) | (0.5176, 0.1842, 0.2982) | 0.524333 | (0.1412, 0.1284, 0.2656) |
| p030 | 6570 | (0.6152, 0.2542, 0.1305) | (0.4330, 0.2404, 0.3266) | 0.426661 | (0.1971, 0.0595, 0.2005) |
| p060 | 2225 | (0.0278, 0.5832, 0.3891) | (0.7059, 0.1944, 0.0998) | 1.356179 | (0.6846, 0.3892, 0.3036) |

## Safe interpretation

This is the grouped-target counterpart of F10. It should reduce repeat-level label noise, but it is still diagnostic and should not be deployed without broader terrain/objective diversity.
