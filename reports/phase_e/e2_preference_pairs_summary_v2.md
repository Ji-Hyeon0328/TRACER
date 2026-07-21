# TRACER Phase-E2 Preference Pairs v0

This is a bootstrap preference-pair dataset built from rollout-level E1 metrics. It is not yet true IRL; it provides weak preference supervision for later Objective Selector training.

- metrics_csv: `datasets/phase_e/e1_rollout_objective_metrics_v2.csv`
- out_csv: `datasets/phase_e/e2_preference_pairs_v2.csv`
- num_pairs: `56`
- min_diff: `0.03`

## Pair counts by preference type

- balanced: `25`
- energy: `3`
- motion: `3`
- stability: `25`

## Pair counts by reset_y

- -0.3: `44`
- 0.0: `10`
- 0.3: `2`

## Top preferred/rejected dimension patterns

- stability: `False/False/True` > `False/False/True` : `6`
- balanced: `False/False/True` > `False/False/True` : `6`
- stability: `False/False/True` > `True/True/True` : `5`
- balanced: `False/False/True` > `True/True/True` : `5`
- stability: `False/False/True` > `True/False/True` : `4`
- balanced: `False/False/True` > `True/False/True` : `4`
- stability: `True/True/True` > `True/True/True` : `3`
- balanced: `True/True/True` > `True/True/True` : `3`
- stability: `True/True/True` > `False/False/True` : `2`
- balanced: `True/True/True` > `False/False/True` : `2`
- motion: `False/False/True` > `False/False/True` : `2`
- stability: `False/False/True` > `False/True/True` : `2`
- energy: `False/False/True` > `False/False/True` : `2`
- balanced: `False/False/True` > `False/True/True` : `2`
- motion: `False/True/True` > `False/False/True` : `1`
- stability: `False/True/True` > `True/True/True` : `1`
- stability: `False/True/True` > `True/False/True` : `1`
- stability: `False/True/True` > `False/False/True` : `1`
- energy: `False/True/True` > `False/False/True` : `1`
- balanced: `False/True/True` > `True/True/True` : `1`

## Example pairs

| pref_type | reset_y | preferred | rejected | diff | pref max_y | rej max_y | pref accept | rej accept |
|---|---:|---|---|---:|---:|---:|---:|---:|
| stability | -0.30 | False/False/True | True/True/True | 0.856 | 0.199 | 0.694 | 0.994 | 0.929 |
| stability | -0.30 | False/False/True | True/False/True | 0.832 | 0.199 | 0.667 | 0.994 | 0.960 |
| stability | -0.30 | False/False/True | True/True/True | 0.651 | 0.321 | 0.694 | 0.993 | 0.929 |
| stability | -0.30 | False/False/True | True/False/True | 0.627 | 0.321 | 0.667 | 0.993 | 0.960 |
| stability | -0.30 | False/False/True | False/False/True | 0.593 | 0.199 | 0.501 | 0.994 | 0.994 |
| stability | -0.30 | False/False/True | False/True/True | 0.546 | 0.199 | 0.505 | 0.994 | 0.994 |
| stability | -0.30 | False/False/True | False/False/True | 0.539 | 0.199 | 0.518 | 0.994 | 0.993 |
| balanced | -0.30 | False/False/True | True/True/True | 0.438 | 0.199 | 0.694 | 0.994 | 0.929 |
| balanced | -0.30 | False/False/True | True/False/True | 0.427 | 0.199 | 0.667 | 0.994 | 0.960 |
| stability | -0.30 | False/False/True | False/False/True | 0.388 | 0.321 | 0.501 | 0.993 | 0.994 |
| stability | -0.30 | False/False/True | False/True/True | 0.341 | 0.321 | 0.505 | 0.993 | 0.994 |
| stability | -0.30 | False/False/True | False/False/True | 0.334 | 0.321 | 0.518 | 0.993 | 0.993 |
| balanced | -0.30 | False/False/True | True/True/True | 0.330 | 0.321 | 0.694 | 0.993 | 0.929 |
| stability | +0.30 | True/True/True | False/False/True | 0.321 | 0.086 | 0.327 | 0.995 | 0.995 |
| balanced | -0.30 | False/False/True | False/False/True | 0.320 | 0.199 | 0.501 | 0.994 | 0.994 |
| balanced | -0.30 | False/False/True | True/False/True | 0.319 | 0.321 | 0.667 | 0.993 | 0.960 |
| stability | -0.30 | False/False/True | True/True/True | 0.317 | 0.518 | 0.694 | 0.993 | 0.929 |
| stability | -0.30 | False/True/True | True/True/True | 0.309 | 0.505 | 0.694 | 0.994 | 0.929 |
| stability | -0.30 | False/False/True | True/False/True | 0.294 | 0.518 | 0.667 | 0.993 | 0.960 |
| stability | +0.00 | True/True/True | True/True/True | 0.292 | 0.196 | 0.417 | 0.994 | 0.994 |
