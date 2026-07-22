# TRACER Phase-J18 Repeat Study Summary v0

- log root: `logs/phase_j/j18_repeat_study_20260722_020336`
- summary csv: `logs/phase_j/j18_repeat_study_20260722_020336/j18_repeat_study_summary_v0.csv`

## Group summary

| mode | n | goals | max_abs_y mean | max_abs_y min | max_abs_y max | mean_abs_y mean |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 6 | 6 | 0.369372 | 0.203091 | 0.534749 | 0.164393 |
| flat_noop | 6 | 6 | 0.256871 | 0.184639 | 0.441479 | 0.111159 |
| upslope_noop | 6 | 5 | 0.611617 | 0.191963 | 2.004607 | 0.204426 |

## Per-run results

| mode | idx | goal | final_x | final_y | max_abs_y | mean_abs_y | j7_accept_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | True | 8.057631 | -0.531846 | 0.534749 | 0.235157 | 0.000000 |
| flat_noop | 1 | True | 8.069722 | -0.208103 | 0.259328 | 0.139709 | 0.000000 |
| upslope_noop | 1 | True | 8.130438 | -0.439264 | 0.439779 | 0.162476 | 0.000000 |
| baseline | 2 | True | 8.075517 | -0.446977 | 0.501532 | 0.260088 | 0.000000 |
| flat_noop | 2 | True | 8.097629 | -0.087113 | 0.230957 | 0.087657 | 0.000000 |
| upslope_noop | 2 | True | 8.113002 | -0.389278 | 0.390388 | 0.197955 | 0.000000 |
| baseline | 3 | True | 8.146807 | -0.064511 | 0.226340 | 0.100554 | 0.000000 |
| flat_noop | 3 | True | 7.975298 | -0.273348 | 0.441479 | 0.200978 | 0.000000 |
| upslope_noop | 3 | True | 8.122429 | 0.048583 | 0.203747 | 0.088702 | 0.000000 |
| baseline | 4 | True | 8.130784 | 0.036054 | 0.307759 | 0.075052 | 0.000000 |
| flat_noop | 4 | True | 7.978492 | -0.186327 | 0.234830 | 0.086030 | 0.000000 |
| upslope_noop | 4 | False | 7.543672 | -1.989548 | 2.004607 | 0.520830 | 0.000000 |
| baseline | 5 | True | 8.134054 | -0.124166 | 0.203091 | 0.067516 | 0.000000 |
| flat_noop | 5 | True | 8.072193 | -0.009618 | 0.189990 | 0.076314 | 0.000000 |
| upslope_noop | 5 | True | 8.047944 | -0.288925 | 0.439220 | 0.190880 | 0.000000 |
| baseline | 6 | True | 8.161075 | -0.262585 | 0.442763 | 0.247991 | 0.000000 |
| flat_noop | 6 | True | 8.014486 | 0.056183 | 0.184639 | 0.076268 | 0.000000 |
| upslope_noop | 6 | True | 8.094752 | -0.103464 | 0.191963 | 0.065714 | 0.000000 |

## Interpretation guide

- If baseline stays low while flat_noop stays low, flat active routing is acceptable only for no-op/conservative commands.
- If upslope_noop remains higher than baseline, upslope active routing or segment timing should remain frozen.
- If baseline itself varies widely, active comparisons need larger N-repeat statistics before changing the action bank.
