# TRACER Phase-K1 Long Baseline vs J19-Profile Summary v0

- log root: `logs/phase_k/k1_long_baseline_vs_j19_profile_20260722_195542`
- summary csv: `datasets/phase_k/k1_long_baseline_vs_j19_profile_summary_v0.csv`
- timeout guard was enabled per rollout.

## Status counts

| mode | status | count |
|---|---|---:|
| baseline | ok | 10 |
| j19_profile | ok | 9 |
| j19_profile | timeout | 1 |

## OK-run group summary

| mode | n | goals | max_abs_y mean | max_abs_y min | max_abs_y max | mean_abs_y mean |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 10 | 10 | 0.365898 | 0.231022 | 0.621932 | 0.157337 |
| j19_profile | 9 | 9 | 0.347311 | 0.122530 | 0.825721 | 0.165474 |

## Per-run results

| status | mode | idx | goal | final_x | final_y | max_abs_y | mean_abs_y | j7 accepted | flat acc | projected | empirical | failsafe |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ok | baseline | 1 | True | 8.100997 | -0.017384 | 0.304067 | 0.127974 | 0 | NA | 0 | 0 | 0 |
| ok | j19_profile | 1 | True | 8.107830 | -0.374646 | 0.398360 | 0.136461 | 432 | 432/434 | 432 | 1735 | 7 |
| ok | baseline | 2 | True | 8.147048 | -0.425051 | 0.428930 | 0.176088 | 0 | NA | 0 | 0 | 0 |
| ok | j19_profile | 2 | True | 8.126465 | 0.310106 | 0.448506 | 0.245969 | 464 | 464/465 | 464 | 1705 | 5 |
| ok | baseline | 3 | True | 8.039850 | 0.190867 | 0.231022 | 0.076465 | 0 | NA | 0 | 0 | 0 |
| ok | j19_profile | 3 | True | 8.041561 | 0.045381 | 0.122530 | 0.056555 | 433 | 433/435 | 433 | 1731 | 7 |
| ok | baseline | 4 | True | 8.211973 | -0.494151 | 0.494151 | 0.269287 | 0 | NA | 0 | 0 | 0 |
| ok | j19_profile | 4 | True | 8.183993 | -0.803437 | 0.825721 | 0.400564 | 441 | 441/443 | 441 | 1725 | 7 |
| ok | baseline | 5 | True | 8.137728 | -0.420962 | 0.476041 | 0.266975 | 0 | NA | 0 | 0 | 0 |
| ok | j19_profile | 5 | True | 8.102249 | -0.061077 | 0.313040 | 0.144577 | 444 | 444/444 | 444 | 1722 | 5 |
| ok | baseline | 6 | True | 8.096336 | 0.241085 | 0.330449 | 0.118779 | 0 | NA | 0 | 0 | 0 |
| ok | j19_profile | 6 | True | 8.138908 | -0.346228 | 0.405720 | 0.297625 | 462 | 462/463 | 462 | 1705 | 7 |
| ok | baseline | 7 | True | 8.150317 | -0.036835 | 0.232221 | 0.094110 | 0 | NA | 0 | 0 | 0 |
| ok | j19_profile | 7 | True | 8.102549 | -0.110004 | 0.256301 | 0.102450 | 446 | 446/448 | 446 | 1721 | 7 |
| ok | baseline | 8 | True | 8.142905 | -0.048030 | 0.240008 | 0.076154 | 0 | NA | 0 | 0 | 0 |
| timeout | j19_profile | 8 | False | nan | nan | nan | nan | 0 | NA | 0 | 0 | 0 |
| ok | baseline | 9 | True | 8.041699 | -0.581550 | 0.621932 | 0.195508 | 0 | NA | 0 | 0 | 0 |
| ok | j19_profile | 9 | True | 8.184127 | -0.054525 | 0.142834 | 0.041393 | 463 | 463/464 | 463 | 1707 | 8 |
| ok | baseline | 10 | True | 8.158976 | -0.286946 | 0.300163 | 0.172027 | 0 | NA | 0 | 0 | 0 |
| ok | j19_profile | 10 | True | 8.178385 | -0.212398 | 0.212785 | 0.063674 | 449 | 449/450 | 449 | 1717 | 6 |

## Interpretation guide

- If J19-profile stays close to baseline, keep it only as a conservative active-routing scaffold.
- If J19-profile is consistently worse, do not promote the deployable active profile.
- If J19-profile is better with high flat acceptance, it becomes the current safe active candidate.
- Timeout rows should be treated as infrastructure failures, not locomotion failures.
