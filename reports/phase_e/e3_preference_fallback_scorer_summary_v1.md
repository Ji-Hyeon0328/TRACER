# TRACER Phase-E3 Preference Fallback Scorer v0

This is a pairwise-logistic bootstrap scorer trained from E2 filtered preference pairs. It is not a true IRL Objective Selector; it is a weak preference-supervised fallback/deployment scorer.

- metrics_csv: `datasets/phase_e/e1_rollout_objective_metrics_v2.csv`
- pairs_csv: `datasets/phase_e/e2_preference_pairs_filtered_v2.csv`
- out_json: `models/phase_e/e3_preference_fallback_scorer_v1.json`
- num_pairs: `32`
- pair_accuracy: `0.875`
- epochs: `4000`
- lr: `0.05`
- l2: `0.001`

## Weights

| feature | weight |
|---|---:|
| bias | 0.000000 |
| reset_y | 0.000000 |
| abs_reset_y | 0.000000 |
| learned_vx | -1.386197 |
| learned_yaw | 1.058413 |
| learned_clearance | 0.000000 |
| abs_reset_y_x_learned_vx | -1.557604 |
| abs_reset_y_x_learned_yaw | -0.824221 |
| abs_reset_y_x_learned_clearance | 0.000000 |
| reset_y_x_learned_vx | 4.521442 |
| reset_y_x_learned_yaw | 3.788059 |
| reset_y_x_learned_clearance | 0.000000 |

## Training history

| epoch | loss | pair_acc |
|---:|---:|---:|
| 0 | 0.693147 | 0.000 |
| 1 | 0.687753 | 0.812 |
| 2 | 0.682504 | 0.812 |
| 5 | 0.667591 | 0.812 |
| 10 | 0.645273 | 0.812 |
| 50 | 0.539534 | 0.812 |
| 100 | 0.486206 | 0.812 |
| 500 | 0.387495 | 0.750 |
| 1000 | 0.337931 | 0.750 |
| 2000 | 0.286100 | 0.875 |
| 3999 | 0.247345 | 0.875 |

## Rollout scores

| idx | reset_y | dims | score | final_y | path_max_abs_y | moving_accept |
|---:|---:|---|---:|---:|---:|---:|
| 4 | +0.30 | True/True/True | 1.4505 | +0.082 | 0.086 | 0.995 |
| 7 | -0.30 | False/False/True | 0.0000 | +0.183 | 0.199 | 0.994 |
| 8 | -0.30 | False/False/True | 0.0000 | -0.259 | 0.321 | 0.993 |
| 9 | -0.30 | False/False/True | 0.0000 | -0.503 | 0.518 | 0.993 |
| 10 | -0.30 | False/False/True | 0.0000 | -0.443 | 0.501 | 0.994 |
| 11 | +0.30 | False/False/True | 0.0000 | -0.008 | 0.327 | 0.995 |
| 12 | +0.00 | False/False/True | 0.0000 | -0.120 | 0.353 | 0.994 |
| 6 | -0.30 | False/True/True | -0.3253 | +0.474 | 0.505 | 0.994 |
| 0 | +0.00 | True/True/True | -0.3278 | -0.140 | 0.196 | 0.994 |
| 1 | +0.00 | True/True/True | -0.3278 | +0.138 | 0.364 | 0.992 |
| 2 | +0.00 | True/True/True | -0.3278 | -0.058 | 0.417 | 0.994 |
| 5 | -0.30 | True/False/True | -3.2099 | +0.664 | 0.667 | 0.960 |
| 3 | -0.30 | True/True/True | -3.5352 | +0.694 | 0.694 | 0.929 |

## Top pair predictions

| pref_type | reset_y | preferred | rejected | model_margin | label_diff | ok |
|---|---:|---|---|---:|---:|---|
| stability | -0.30 | False/False/True | True/True/True | 3.5352 | 0.8555 | True |
| stability | -0.30 | False/False/True | True/True/True | 3.5352 | 0.6506 | True |
| stability | -0.30 | False/False/True | True/True/True | 3.5352 | 0.3168 | True |
| stability | -0.30 | False/False/True | True/True/True | 3.5352 | 0.2628 | True |
| balanced | -0.30 | False/False/True | True/True/True | 3.5352 | 0.4384 | True |
| balanced | -0.30 | False/False/True | True/True/True | 3.5352 | 0.3302 | True |
| balanced | -0.30 | False/False/True | True/True/True | 3.5352 | 0.1594 | True |
| balanced | -0.30 | False/False/True | True/True/True | 3.5352 | 0.1188 | True |
| stability | -0.30 | False/True/True | True/True/True | 3.2099 | 0.3092 | True |
| stability | -0.30 | False/False/True | True/False/True | 3.2099 | 0.8323 | True |
| stability | -0.30 | False/False/True | True/False/True | 3.2099 | 0.6273 | True |
| stability | -0.30 | False/False/True | True/False/True | 3.2099 | 0.2936 | True |
| stability | -0.30 | False/False/True | True/False/True | 3.2099 | 0.2396 | True |
| balanced | -0.30 | False/True/True | True/True/True | 3.2099 | 0.1595 | True |
| balanced | -0.30 | False/False/True | True/False/True | 3.2099 | 0.4275 | True |
| balanced | -0.30 | False/False/True | True/False/True | 3.2099 | 0.3193 | True |
| balanced | -0.30 | False/False/True | True/False/True | 3.2099 | 0.1485 | True |
| balanced | -0.30 | False/False/True | True/False/True | 3.2099 | 0.1079 | True |
| stability | -0.30 | False/True/True | True/False/True | 2.8846 | 0.2860 | True |
| balanced | -0.30 | False/True/True | True/False/True | 2.8846 | 0.1487 | True |
| stability | +0.30 | True/True/True | False/False/True | 1.4505 | 0.3205 | True |
| balanced | +0.30 | True/True/True | False/False/True | 1.4505 | 0.1645 | True |
| stability | +0.00 | True/True/True | False/False/True | -0.3278 | 0.2301 | False |
| stability | +0.00 | False/False/True | True/True/True | 0.3278 | 0.0615 | True |
| balanced | +0.00 | True/True/True | False/False/True | -0.3278 | 0.1180 | False |
| balanced | +0.00 | False/False/True | True/True/True | 0.3278 | 0.0373 | True |
| stability | -0.30 | False/False/True | False/True/True | 0.3253 | 0.5463 | True |
| stability | -0.30 | False/False/True | False/True/True | 0.3253 | 0.3414 | True |
| stability | -0.30 | False/True/True | False/False/True | -0.3253 | 0.0464 | False |
| balanced | -0.30 | False/False/True | False/True/True | 0.3253 | 0.2788 | True |
