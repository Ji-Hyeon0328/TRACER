# TRACER Phase-F10 Runtime-Safe Beta Calibrator v0

This trains a diagnostic beta calibrator using only runtime-available D7/log features, excluding true-metric target leakage.

- input: `datasets/phase_f/f7_d7_true_metric_calibration_dataset_v0.csv`
- model: `models/phase_f/f10_runtime_safe_beta_calibrator_ridge_v0.json`
- rows: `6677`
- features: `35`
- alpha: `1.0`

## Train error

| beta | RMSE | MAE |
|---|---:|---:|
| motion | 0.145251 | 0.110526 |
| stability | 0.012290 | 0.010801 |
| energy | 0.143871 | 0.108919 |

## Per-condition train-fit mean

| tag | rows | target beta | predicted beta | L1 mean error |
|---|---:|---|---|---:|
| clean | 2226 | (0.3333, 0.3333, 0.3333) | (0.3213, 0.3171, 0.3616) | 0.056504 |
| m030 | 2225 | (0.0441, 0.3104, 0.6456) | (0.1640, 0.3153, 0.5206) | 0.249881 |
| p030 | 2226 | (0.6549, 0.2985, 0.0466) | (0.5531, 0.3075, 0.1395) | 0.203616 |

## Leave-one-tag-out diagnostic

| heldout | rows | target mean | predicted mean | mean L1 error | RMSE |
|---|---:|---|---|---:|---|
| clean | 2226 | (0.3333, 0.3333, 0.3333) | (0.3150, 0.3051, 0.3799) | 0.259223 | (0.1313, 0.0283, 0.1358) |
| m030 | 2225 | (0.0441, 0.3104, 0.6456) | (0.2484, 0.3406, 0.4110) | 0.486057 | (0.2745, 0.0351, 0.2875) |
| p030 | 2226 | (0.6549, 0.2985, 0.0466) | (0.4102, 0.3394, 0.2504) | 0.489433 | (0.2795, 0.0423, 0.2505) |

## Safe interpretation

This is stricter than F8 because it removes true-metric leakage features. If leave-one-tag-out is weak, that means the current three-condition dataset is insufficient for a deployable beta calibrator. Use this as a diagnostic before expanding F3 data.
