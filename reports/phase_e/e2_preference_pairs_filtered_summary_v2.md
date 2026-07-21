# TRACER Phase-E2 Preference Pairs v1 Filtered

This filters E2 v0 into cross-policy preference pairs for bootstrap policy/objective selection.

- in_csv: `datasets/phase_e/e2_preference_pairs_v2.csv`
- out_csv: `datasets/phase_e/e2_preference_pairs_filtered_v2.csv`
- kept_pairs: `32`
- dropped_same_dims: `18`
- dropped_type: `6`
- keep_types: `balanced,stability`

## Counts by preference type

- balanced: `16`
- stability: `16`

## Counts by reset_y

- -0.3: `26`
- 0.0: `4`
- 0.3: `2`

## Dimension preference patterns

- `False/False/True` > `True/True/True` : `10`
- `False/False/True` > `True/False/True` : `8`
- `True/True/True` > `False/False/True` : `4`
- `False/False/True` > `False/True/True` : `4`
- `False/True/True` > `True/True/True` : `2`
- `False/True/True` > `True/False/True` : `2`
- `False/True/True` > `False/False/True` : `2`

## Top filtered pairs

| pref_type | reset_y | preferred | rejected | diff | pref max_y | rej max_y | pref accept | rej accept |
|---|---:|---|---|---:|---:|---:|---:|---:|
| stability | -0.30 | False/False/True | True/True/True | 0.856 | 0.199 | 0.694 | 0.994 | 0.929 |
| stability | -0.30 | False/False/True | True/False/True | 0.832 | 0.199 | 0.667 | 0.994 | 0.960 |
| stability | -0.30 | False/False/True | True/True/True | 0.651 | 0.321 | 0.694 | 0.993 | 0.929 |
| stability | -0.30 | False/False/True | True/False/True | 0.627 | 0.321 | 0.667 | 0.993 | 0.960 |
| stability | -0.30 | False/False/True | False/True/True | 0.546 | 0.199 | 0.505 | 0.994 | 0.994 |
| balanced | -0.30 | False/False/True | True/True/True | 0.438 | 0.199 | 0.694 | 0.994 | 0.929 |
| balanced | -0.30 | False/False/True | True/False/True | 0.427 | 0.199 | 0.667 | 0.994 | 0.960 |
| stability | -0.30 | False/False/True | False/True/True | 0.341 | 0.321 | 0.505 | 0.993 | 0.994 |
| balanced | -0.30 | False/False/True | True/True/True | 0.330 | 0.321 | 0.694 | 0.993 | 0.929 |
| stability | +0.30 | True/True/True | False/False/True | 0.321 | 0.086 | 0.327 | 0.995 | 0.995 |
| balanced | -0.30 | False/False/True | True/False/True | 0.319 | 0.321 | 0.667 | 0.993 | 0.960 |
| stability | -0.30 | False/False/True | True/True/True | 0.317 | 0.518 | 0.694 | 0.993 | 0.929 |
| stability | -0.30 | False/True/True | True/True/True | 0.309 | 0.505 | 0.694 | 0.994 | 0.929 |
| stability | -0.30 | False/False/True | True/False/True | 0.294 | 0.518 | 0.667 | 0.993 | 0.960 |
| stability | -0.30 | False/True/True | True/False/True | 0.286 | 0.505 | 0.667 | 0.994 | 0.960 |
| balanced | -0.30 | False/False/True | False/True/True | 0.279 | 0.199 | 0.505 | 0.994 | 0.994 |
| stability | -0.30 | False/False/True | True/True/True | 0.263 | 0.501 | 0.694 | 0.994 | 0.929 |
| stability | -0.30 | False/False/True | True/False/True | 0.240 | 0.501 | 0.667 | 0.994 | 0.960 |
| stability | +0.00 | True/True/True | False/False/True | 0.230 | 0.196 | 0.353 | 0.994 | 0.994 |
| balanced | -0.30 | False/False/True | False/True/True | 0.171 | 0.321 | 0.505 | 0.993 | 0.994 |
