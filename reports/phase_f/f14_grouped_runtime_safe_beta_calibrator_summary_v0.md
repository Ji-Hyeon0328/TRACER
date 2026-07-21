# TRACER Phase-F14 Grouped Runtime-Safe Beta Calibrator v0

This trains a diagnostic beta calibrator using runtime-available D7/log features and Phase-F12 grouped true-metric beta targets.

- input: `datasets/phase_f/f13_d7_grouped_beta_calibration_dataset_v0.csv`
- model: `models/phase_f/f14_grouped_runtime_safe_beta_calibrator_ridge_v0.json`
- rows: `28513`
- features: `36`
- alpha: `1.0`

## Train error

| beta | RMSE | MAE |
|---|---:|---:|
| motion | 0.140532 | 0.118945 |
| stability | 0.079948 | 0.057744 |
| energy | 0.151235 | 0.128167 |

## Per-condition train-fit mean

| tag | rows | target beta | predicted beta | L1 mean error |
|---|---:|---|---|---:|
| clean | 6672 | (0.3679, 0.2194, 0.4127) | (0.4447, 0.2205, 0.3348) | 0.155789 |
| m015 | 2124 | (0.6278, 0.2262, 0.1460) | (0.4922, 0.2328, 0.2750) | 0.271172 |
| m030 | 6572 | (0.3334, 0.1478, 0.5188) | (0.4375, 0.2108, 0.3517) | 0.334298 |
| m060 | 2226 | (0.5482, 0.1360, 0.3158) | (0.4697, 0.2202, 0.3101) | 0.168431 |
| p015 | 2124 | (0.6354, 0.3254, 0.0392) | (0.4732, 0.2255, 0.3013) | 0.524160 |
| p030 | 6570 | (0.5884, 0.2722, 0.1394) | (0.4585, 0.2572, 0.2843) | 0.289733 |
| p060 | 2225 | (0.0278, 0.5832, 0.3891) | (0.2479, 0.4357, 0.3164) | 0.440276 |

## Leave-one-tag-out diagnostic

| heldout | rows | target mean | predicted mean | mean L1 error | RMSE |
|---|---:|---|---|---:|---|
| clean | 6672 | (0.3679, 0.2194, 0.4127) | (0.5264, 0.2201, 0.2535) | 0.353649 | (0.1708, 0.0435, 0.1730) |
| m015 | 2124 | (0.6278, 0.2262, 0.1460) | (0.4756, 0.2342, 0.2901) | 0.312350 | (0.1674, 0.0203, 0.1603) |
| m030 | 6572 | (0.3334, 0.1478, 0.5188) | (0.4917, 0.2419, 0.2664) | 0.507300 | (0.1723, 0.1014, 0.2693) |
| m060 | 2226 | (0.5482, 0.1360, 0.3158) | (0.4563, 0.2330, 0.3107) | 0.213034 | (0.0948, 0.0991, 0.0280) |
| p015 | 2124 | (0.6354, 0.3254, 0.0392) | (0.4449, 0.2068, 0.3482) | 0.618069 | (0.2002, 0.1199, 0.3159) |
| p030 | 6570 | (0.5884, 0.2722, 0.1394) | (0.3947, 0.2538, 0.3516) | 0.455309 | (0.2021, 0.0589, 0.2160) |
| p060 | 2225 | (0.0278, 0.5832, 0.3891) | (0.5971, 0.2688, 0.1341) | 1.138583 | (0.5729, 0.3164, 0.2732) |

## Safe interpretation

This is the grouped-target counterpart of F10. It should reduce repeat-level label noise, but it is still diagnostic and should not be deployed without broader terrain/objective diversity.
