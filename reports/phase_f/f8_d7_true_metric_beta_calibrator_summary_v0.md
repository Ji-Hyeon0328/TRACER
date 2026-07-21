# TRACER Phase-F8 D7 True-Metric Beta Calibrator v0

This trains a ridge calibration model from D7 runtime features to Phase-F6 true-metric beta target seeds.

- input csv: `datasets/phase_f/f7_d7_true_metric_calibration_dataset_v0.csv`
- model json: `models/phase_f/f8_d7_true_metric_beta_calibrator_ridge_v0.json`
- rows: `6677`
- features: `42`
- alpha: `1.0`

## Train error

| beta | RMSE | MAE |
|---|---:|---:|
| motion | 0.000018 | 0.000014 |
| stability | 0.000001 | 0.000001 |
| energy | 0.000018 | 0.000014 |

## Per-condition mean prediction

| tag | rows | target beta | predicted beta | abs error |
|---|---:|---|---|---|
| clean | 2226 | (0.3333, 0.3333, 0.3333) | (0.3333, 0.3333, 0.3333) | (0.0000, 0.0000, 0.0000) |
| m030 | 2225 | (0.0441, 0.3104, 0.6456) | (0.0441, 0.3104, 0.6456) | (0.0000, 0.0000, 0.0000) |
| p030 | 2226 | (0.6549, 0.2985, 0.0466) | (0.6549, 0.2985, 0.0466) | (0.0000, 0.0000, 0.0000) |

## Leave-one-tag-out diagnostic

| heldout | rows | target mean | predicted mean | RMSE |
|---|---:|---|---|---|
| clean | 2226 | (0.3333, 0.3333, 0.3333) | (0.3259, 0.3049, 0.3692) | (0.0074, 0.0284, 0.0358) |
| m030 | 2225 | (0.0441, 0.3104, 0.6456) | (0.7152, 0.2848, 0.0000) | (0.6711, 0.0255, 0.6456) |
| p030 | 2226 | (0.6549, 0.2985, 0.0466) | (0.0000, 0.2661, 0.7339) | (0.6549, 0.0324, 0.6873) |

## Safe interpretation

This is a diagnostic calibration model, not a deployable replacement for D7. The dataset has many timestep rows but only three source rollout conditions, so train error can be optimistic. Use this to quantify the gap between existing D7 runtime beta behavior and true-metric beta seeds.
