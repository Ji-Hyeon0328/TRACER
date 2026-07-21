# TRACER Phase-E3 Preference Fallback Scorer v0

This is a pairwise-logistic bootstrap scorer trained from E2 filtered preference pairs. It is not a true IRL Objective Selector; it is a weak preference-supervised fallback/deployment scorer.

- metrics_csv: `datasets/phase_e/e1_rollout_objective_metrics_v1.csv`
- pairs_csv: `datasets/phase_e/e2_preference_pairs_filtered_v1.csv`
- out_json: `models/phase_e/e3_preference_fallback_scorer_v0.json`
- num_pairs: `26`
- pair_accuracy: `0.923`
- epochs: `4000`
- lr: `0.05`
- l2: `0.001`

## Weights

| feature | weight |
|---|---:|
| bias | 0.000000 |
| reset_y | 0.000000 |
| abs_reset_y | 0.000000 |
| learned_vx | -4.283021 |
| learned_yaw | -0.587233 |
| learned_clearance | 0.000000 |
| abs_reset_y_x_learned_vx | -1.284906 |
| abs_reset_y_x_learned_yaw | -0.176170 |
| abs_reset_y_x_learned_clearance | 0.000000 |
| reset_y_x_learned_vx | 1.284906 |
| reset_y_x_learned_yaw | 0.176170 |
| reset_y_x_learned_clearance | 0.000000 |

## Training history

| epoch | loss | pair_acc |
|---:|---:|---:|
| 0 | 0.693147 | 0.000 |
| 1 | 0.683091 | 0.923 |
| 2 | 0.673301 | 0.923 |
| 5 | 0.645464 | 0.923 |
| 10 | 0.603755 | 0.923 |
| 50 | 0.405864 | 0.923 |
| 100 | 0.307148 | 0.923 |
| 500 | 0.182992 | 0.923 |
| 1000 | 0.164941 | 0.923 |
| 2000 | 0.156030 | 0.923 |
| 3999 | 0.151717 | 0.923 |

## Rollout scores

| idx | reset_y | dims | score | final_y | path_max_abs_y | moving_accept |
|---:|---:|---|---:|---:|---:|---:|
| 7 | -0.30 | False/False/True | 0.0000 | +0.183 | 0.199 | 0.994 |
| 8 | -0.30 | False/False/True | 0.0000 | -0.259 | 0.321 | 0.993 |
| 9 | -0.30 | False/False/True | 0.0000 | -0.503 | 0.518 | 0.993 |
| 10 | -0.30 | False/False/True | 0.0000 | -0.443 | 0.501 | 0.994 |
| 6 | -0.30 | False/True/True | -0.6929 | +0.474 | 0.505 | 0.994 |
| 0 | +0.00 | True/True/True | -4.8703 | -0.140 | 0.196 | 0.994 |
| 1 | +0.00 | True/True/True | -4.8703 | +0.138 | 0.364 | 0.992 |
| 2 | +0.00 | True/True/True | -4.8703 | -0.058 | 0.417 | 0.994 |
| 4 | +0.30 | True/True/True | -4.8703 | +0.082 | 0.086 | 0.995 |
| 5 | -0.30 | True/False/True | -5.0540 | +0.664 | 0.667 | 0.960 |
| 3 | -0.30 | True/True/True | -5.7469 | +0.694 | 0.694 | 0.929 |

## Top pair predictions

| pref_type | reset_y | preferred | rejected | model_margin | label_diff | ok |
|---|---:|---|---|---:|---:|---|
| stability | -0.30 | False/False/True | True/True/True | 5.7469 | 0.8555 | True |
| stability | -0.30 | False/False/True | True/True/True | 5.7469 | 0.6506 | True |
| stability | -0.30 | False/False/True | True/True/True | 5.7469 | 0.3168 | True |
| stability | -0.30 | False/False/True | True/True/True | 5.7469 | 0.2628 | True |
| balanced | -0.30 | False/False/True | True/True/True | 5.7469 | 0.4384 | True |
| balanced | -0.30 | False/False/True | True/True/True | 5.7469 | 0.3302 | True |
| balanced | -0.30 | False/False/True | True/True/True | 5.7469 | 0.1594 | True |
| balanced | -0.30 | False/False/True | True/True/True | 5.7469 | 0.1188 | True |
| stability | -0.30 | False/False/True | True/False/True | 5.0540 | 0.8323 | True |
| stability | -0.30 | False/False/True | True/False/True | 5.0540 | 0.6273 | True |
| stability | -0.30 | False/False/True | True/False/True | 5.0540 | 0.2936 | True |
| stability | -0.30 | False/False/True | True/False/True | 5.0540 | 0.2396 | True |
| balanced | -0.30 | False/False/True | True/False/True | 5.0540 | 0.4275 | True |
| balanced | -0.30 | False/False/True | True/False/True | 5.0540 | 0.3193 | True |
| balanced | -0.30 | False/False/True | True/False/True | 5.0540 | 0.1485 | True |
| balanced | -0.30 | False/False/True | True/False/True | 5.0540 | 0.1079 | True |
| stability | -0.30 | False/True/True | True/True/True | 5.0540 | 0.3092 | True |
| balanced | -0.30 | False/True/True | True/True/True | 5.0540 | 0.1595 | True |
| stability | -0.30 | False/True/True | True/False/True | 4.3610 | 0.2860 | True |
| balanced | -0.30 | False/True/True | True/False/True | 4.3610 | 0.1487 | True |
| stability | -0.30 | False/False/True | False/True/True | 0.6929 | 0.5463 | True |
| stability | -0.30 | False/False/True | False/True/True | 0.6929 | 0.3414 | True |
| stability | -0.30 | False/True/True | False/False/True | -0.6929 | 0.0464 | False |
| balanced | -0.30 | False/False/True | False/True/True | 0.6929 | 0.2788 | True |
| balanced | -0.30 | False/False/True | False/True/True | 0.6929 | 0.1707 | True |
| balanced | -0.30 | False/True/True | False/False/True | -0.6929 | 0.0407 | False |
