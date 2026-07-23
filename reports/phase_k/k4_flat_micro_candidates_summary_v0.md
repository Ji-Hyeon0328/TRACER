# TRACER Phase-K4 Flat Micro-Action Candidates Summary v0

- log root: `logs/phase_k/k4_flat_micro_candidates_20260723_141518`
- summary csv: `datasets/phase_k/k4_flat_micro_candidates_summary_v0.csv`
- timeout guard was enabled per rollout.

## Status counts

| mode | status | count |
|---|---|---:|
| baseline | ok | 6 |
| flat_clear03 | ok | 6 |
| flat_noop | ok | 6 |
| flat_slow03 | ok | 6 |
| flat_slow03_clear03 | ok | 6 |

## OK-run group summary

| mode | n | goals | max_abs_y mean | max_abs_y min | max_abs_y max | mean_abs_y mean | j7 accepted mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 6 | 6 | 0.445419 | 0.165260 | 0.751836 | 0.171278 | 0.00 |
| flat_clear03 | 6 | 0 | 0.146234 | 0.070704 | 0.203231 | 0.073035 | 0.00 |
| flat_noop | 6 | 0 | 0.141203 | 0.068845 | 0.319740 | 0.066033 | 0.00 |
| flat_slow03 | 6 | 0 | nan | 0.056569 | 0.142685 | nan | 0.00 |
| flat_slow03_clear03 | 6 | 0 | 0.139324 | 0.068265 | 0.256840 | 0.063410 | 0.00 |

## Per-run results

| status | mode | idx | goal | final_x | final_y | max_abs_y | mean_abs_y | j7 accepted | flat acc | projected | empirical | failsafe |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ok | baseline | 1 | True | 8.093605 | -0.396888 | 0.397087 | 0.117961 | 0 | NA | 0 | 0 | 0 |
| ok | flat_noop | 1 | False | 0.157810 | -0.141129 | 0.148897 | 0.077251 | 0 | 0/655 | 0 | 0 | 666 |
| ok | flat_slow03 | 1 | False | 0.092904 | -0.080555 | 0.080555 | 0.035289 | 0 | 0/662 | 0 | 0 | 663 |
| ok | flat_clear03 | 1 | False | 0.087363 | -0.203231 | 0.203231 | 0.084047 | 0 | 0/666 | 0 | 0 | 667 |
| ok | flat_slow03_clear03 | 1 | False | 0.200567 | -0.122973 | 0.122973 | 0.033487 | 0 | 0/655 | 0 | 0 | 659 |
| ok | baseline | 2 | True | 8.060887 | -0.374541 | 0.383029 | 0.117531 | 0 | NA | 0 | 0 | 0 |
| ok | flat_noop | 2 | False | 0.146736 | -0.027188 | 0.068845 | 0.023798 | 0 | 0/654 | 0 | 0 | 661 |
| ok | flat_slow03 | 2 | False | 0.078941 | -0.024638 | 0.125204 | 0.069959 | 0 | 0/654 | 0 | 0 | 654 |
| ok | flat_clear03 | 2 | False | 0.094232 | -0.169481 | 0.171163 | 0.090468 | 0 | 0/655 | 0 | 0 | 662 |
| ok | flat_slow03_clear03 | 2 | False | 0.128061 | -0.019154 | 0.068265 | 0.034046 | 0 | 0/656 | 0 | 0 | 663 |
| ok | baseline | 3 | True | 8.142710 | -0.508898 | 0.560099 | 0.185874 | 0 | NA | 0 | 0 | 0 |
| ok | flat_noop | 3 | False | 0.092140 | -0.060383 | 0.071897 | 0.048511 | 0 | 0/654 | 0 | 0 | 658 |
| ok | flat_slow03 | 3 | False | 0.124451 | -0.010662 | 0.056569 | 0.018534 | 0 | 0/655 | 0 | 0 | 662 |
| ok | flat_clear03 | 3 | False | 0.077695 | -0.111330 | 0.135501 | 0.065170 | 0 | 0/656 | 0 | 0 | 660 |
| ok | flat_slow03_clear03 | 3 | False | 0.118555 | -0.105928 | 0.107178 | 0.042973 | 0 | 0/660 | 0 | 0 | 664 |
| ok | baseline | 4 | True | 8.163793 | 0.102613 | 0.165260 | 0.063852 | 0 | NA | 0 | 0 | 0 |
| ok | flat_noop | 4 | False | 0.032074 | -0.319028 | 0.319740 | 0.171840 | 0 | 0/657 | 0 | 0 | 660 |
| ok | flat_slow03 | 4 | False | 0.168037 | -0.114238 | 0.142685 | 0.070951 | 0 | 0/655 | 0 | 0 | 659 |
| ok | flat_clear03 | 4 | False | 0.029584 | -0.088088 | 0.117749 | 0.052646 | 0 | 0/653 | 0 | 0 | 653 |
| ok | flat_slow03_clear03 | 4 | False | 0.106886 | -0.013979 | 0.205665 | 0.104629 | 0 | 0/654 | 0 | 0 | 658 |
| ok | baseline | 5 | True | 8.038273 | -0.734729 | 0.751836 | 0.321166 | 0 | NA | 0 | 0 | 0 |
| ok | flat_noop | 5 | False | 0.080867 | -0.122261 | 0.122261 | 0.035096 | 0 | 0/656 | 0 | 0 | 660 |
| ok | flat_slow03 | 5 | False | nan | nan | nan | nan | 0 | NA | 0 | 0 | 660 |
| ok | flat_clear03 | 5 | False | 0.040150 | 0.036077 | 0.070704 | 0.035895 | 0 | 0/655 | 0 | 0 | 655 |
| ok | flat_slow03_clear03 | 5 | False | 0.073167 | -0.032829 | 0.075026 | 0.037840 | 0 | 0/654 | 0 | 0 | 661 |
| ok | baseline | 6 | True | 8.228708 | -0.346296 | 0.415204 | 0.221286 | 0 | NA | 0 | 0 | 0 |
| ok | flat_noop | 6 | False | 0.036765 | -0.113672 | 0.115575 | 0.039703 | 0 | 0/656 | 0 | 0 | 663 |
| ok | flat_slow03 | 6 | False | 0.039449 | -0.086370 | 0.089245 | 0.047916 | 0 | 0/656 | 0 | 0 | 660 |
| ok | flat_clear03 | 6 | False | 0.086425 | -0.152364 | 0.179055 | 0.109982 | 0 | 0/660 | 0 | 0 | 671 |
| ok | flat_slow03_clear03 | 6 | False | 0.123104 | -0.240894 | 0.256840 | 0.127485 | 0 | 0/655 | 0 | 0 | 659 |

## Interpretation guide

- A candidate is promising only if it reaches all goals and improves both max_abs_y and mean_abs_y over baseline.
- Timeout rows are infrastructure failures, not locomotion failures.
- If no candidate beats baseline, keep active routing as scaffold/debug and move to learned/data-backed selector design.
