# TRACER Phase-F9 D7 vs True-Metric Beta Comparison v0

This report compares existing D7 runtime beta outputs against Phase-F6 true-metric beta target seeds, and also includes the Phase-F8 train-fit calibrated beta.

- input dataset: `datasets/phase_f/f7_d7_true_metric_calibration_dataset_v0.csv`
- F8 model: `models/phase_f/f8_d7_true_metric_beta_calibrator_ridge_v0.json`
- output csv: `datasets/phase_f/f9_d7_vs_true_metric_beta_comparison_v0.csv`
- rows: `6677`

## Per-condition beta comparison

| tag | method | rows | target beta | predicted beta | mean L1 error | RMSE |
|---|---|---:|---|---|---:|---|
| clean | d7_pred | 2226 | (0.3333, 0.3333, 0.3333) | (0.3257, 0.4485, 0.2259) | 0.376856 | (0.1181, 0.1826, 0.1164) |
| clean | d7_raw | 2226 | (0.3333, 0.3333, 0.3333) | (0.3317, 0.4455, 0.2228) | 0.385564 | (0.1253, 0.1850, 0.1181) |
| clean | d7_prior | 2226 | (0.3333, 0.3333, 0.3333) | (0.3016, 0.4604, 0.2381) | 0.342034 | (0.0946, 0.1787, 0.1120) |
| clean | d7_actual | 2226 | (0.3333, 0.3333, 0.3333) | (0.3017, 0.4603, 0.2381) | 0.341989 | (0.0946, 0.1786, 0.1120) |
| clean | f8_calibrated_trainfit | 2226 | (0.3333, 0.3333, 0.3333) | (0.3333, 0.3333, 0.3333) | 0.000019 | (0.0000, 0.0000, 0.0000) |
| m030 | d7_pred | 2225 | (0.0441, 0.3104, 0.6456) | (0.3105, 0.4765, 0.2130) | 0.903546 | (0.2976, 0.2353, 0.4353) |
| m030 | d7_raw | 2225 | (0.0441, 0.3104, 0.6456) | (0.3127, 0.4802, 0.2071) | 0.918772 | (0.3049, 0.2471, 0.4410) |
| m030 | d7_prior | 2225 | (0.0441, 0.3104, 0.6456) | (0.3018, 0.4617, 0.2365) | 0.842639 | (0.2725, 0.1971, 0.4133) |
| m030 | d7_actual | 2225 | (0.0441, 0.3104, 0.6456) | (0.3019, 0.4616, 0.2365) | 0.842694 | (0.2727, 0.1970, 0.4133) |
| m030 | f8_calibrated_trainfit | 2225 | (0.0441, 0.3104, 0.6456) | (0.0441, 0.3104, 0.6456) | 0.000035 | (0.0000, 0.0000, 0.0000) |
| p030 | d7_pred | 2226 | (0.6549, 0.2985, 0.0466) | (0.3420, 0.4212, 0.2368) | 0.661128 | (0.3320, 0.1772, 0.1942) |
| p030 | d7_raw | 2226 | (0.6549, 0.2985, 0.0466) | (0.3525, 0.4106, 0.2369) | 0.644020 | (0.3243, 0.1723, 0.1934) |
| p030 | d7_prior | 2226 | (0.6549, 0.2985, 0.0466) | (0.2997, 0.4637, 0.2365) | 0.729559 | (0.3660, 0.2075, 0.1991) |
| p030 | d7_actual | 2226 | (0.6549, 0.2985, 0.0466) | (0.2998, 0.4636, 0.2365) | 0.729378 | (0.3659, 0.2075, 0.1991) |
| p030 | f8_calibrated_trainfit | 2226 | (0.6549, 0.2985, 0.0466) | (0.6549, 0.2985, 0.0466) | 0.000032 | (0.0000, 0.0000, 0.0000) |

## Safe interpretation

- `d7_pred`, `d7_raw`, `d7_prior`, and `d7_actual` show how the existing D7 runtime beta behavior differs from true-metric beta seeds.
- `f8_calibrated_trainfit` is included only as an upper-bound diagnostic. It should not be deployed because F8 is trained on only three source rollout conditions and its leave-one-tag-out diagnostic was weak.
- This report is suitable for calibration evidence, not for claiming a completed IRL Objective Selector.
