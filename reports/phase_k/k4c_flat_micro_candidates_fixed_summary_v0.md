# TRACER Phase-K4 Flat Micro-Action Candidates Summary v0

- log root: `logs/phase_k/k4c_flat_micro_candidates_fixed_20260723_164401`
- summary csv: `datasets/phase_k/k4c_flat_micro_candidates_fixed_summary_v0.csv`
- timeout guard was enabled per rollout.

## Status counts

| mode | status | count |
|---|---|---:|
| baseline | ok | 6 |
| flat_clear03 | ok | 6 |
| flat_noop | nonzero_exit | 1 |
| flat_noop | ok | 5 |
| flat_slow03 | ok | 6 |
| flat_slow03_clear03 | ok | 6 |

## OK-run group summary

| mode | n | goals | max_abs_y mean | max_abs_y min | max_abs_y max | mean_abs_y mean | j7 accepted mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 6 | 6 | 0.335126 | 0.199063 | 0.483797 | 0.152324 | 0.00 |
| flat_clear03 | 6 | 6 | 0.399815 | 0.197967 | 0.563978 | 0.223024 | 446.33 |
| flat_noop | 5 | 5 | 0.396019 | 0.267682 | 0.454667 | 0.174589 | 447.00 |
| flat_slow03 | 6 | 6 | 0.329978 | 0.174245 | 0.515005 | 0.157207 | 460.00 |
| flat_slow03_clear03 | 6 | 6 | 0.387286 | 0.181762 | 0.769346 | 0.179587 | 464.67 |

## Per-run results

| status | mode | idx | goal | final_x | final_y | max_abs_y | mean_abs_y | j7 accepted | flat acc | projected | empirical | failsafe |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ok | baseline | 1 | True | 8.141177 | -0.060489 | 0.483797 | 0.220794 | 0 | NA | 0 | 0 | 0 |
| ok | flat_noop | 1 | True | 8.041636 | -0.189066 | 0.413035 | 0.215599 | 435 | 435/436 | 435 | 1734 | 6 |
| ok | flat_slow03 | 1 | True | 8.084136 | -0.059845 | 0.237615 | 0.103612 | 484 | 484/485 | 484 | 1786 | 8 |
| ok | flat_clear03 | 1 | True | 8.208428 | -0.342733 | 0.476883 | 0.276378 | 453 | 453/454 | 453 | 1719 | 8 |
| ok | flat_slow03_clear03 | 1 | True | 7.968275 | -0.171936 | 0.181762 | 0.095139 | 462 | 462/463 | 462 | 1810 | 7 |
| ok | baseline | 2 | True | 8.135780 | -0.193515 | 0.422743 | 0.256692 | 0 | NA | 0 | 0 | 0 |
| ok | flat_noop | 2 | True | 8.189947 | -0.250733 | 0.267682 | 0.107338 | 436 | 436/437 | 436 | 1729 | 7 |
| ok | flat_slow03 | 2 | True | 8.109769 | -0.243312 | 0.515005 | 0.246896 | 438 | 438/438 | 438 | 1740 | 7 |
| ok | flat_clear03 | 2 | True | 8.142377 | -0.138497 | 0.197967 | 0.093569 | 443 | 443/443 | 443 | 1723 | 6 |
| ok | flat_slow03_clear03 | 2 | True | 8.149764 | -0.213440 | 0.372442 | 0.227965 | 485 | 485/485 | 485 | 1781 | 7 |
| ok | baseline | 3 | True | 8.090573 | -0.033035 | 0.410366 | 0.183473 | 0 | NA | 0 | 0 | 0 |
| ok | flat_noop | 3 | True | 8.232712 | -0.435010 | 0.443570 | 0.230737 | 458 | 458/460 | 458 | 1713 | 7 |
| ok | flat_slow03 | 3 | True | 7.950432 | -0.167669 | 0.174245 | 0.076390 | 471 | 471/472 | 471 | 1694 | 8 |
| ok | flat_clear03 | 3 | True | 8.117569 | -0.309984 | 0.409428 | 0.226365 | 454 | 454/455 | 454 | 1724 | 7 |
| ok | flat_slow03_clear03 | 3 | True | 8.153760 | 0.207354 | 0.241698 | 0.097864 | 474 | 474/475 | 474 | 1799 | 8 |
| ok | baseline | 4 | True | 8.041471 | 0.244558 | 0.244558 | 0.065498 | 0 | NA | 0 | 0 | 0 |
| nonzero_exit | flat_noop | 4 | False | nan | nan | nan | nan | 0 | NA | 0 | 0 | 0 |
| ok | flat_slow03 | 4 | True | 8.203701 | 0.062147 | 0.273044 | 0.122430 | 447 | 447/448 | 447 | 1823 | 6 |
| ok | flat_clear03 | 4 | True | 8.012594 | -0.499449 | 0.550439 | 0.316740 | 446 | 446/447 | 446 | 1719 | 7 |
| ok | flat_slow03_clear03 | 4 | True | 7.992806 | -0.565839 | 0.769346 | 0.380733 | 446 | 446/447 | 446 | 1827 | 7 |
| ok | baseline | 5 | True | 8.185383 | -0.124848 | 0.250232 | 0.117853 | 0 | NA | 0 | 0 | 0 |
| ok | flat_noop | 5 | True | 8.128056 | -0.423753 | 0.454667 | 0.189945 | 452 | 452/453 | 452 | 1814 | 7 |
| ok | flat_slow03 | 5 | True | 8.191953 | -0.370798 | 0.462906 | 0.261107 | 462 | 462/463 | 462 | 1804 | 7 |
| ok | flat_clear03 | 5 | True | 8.123902 | 0.186756 | 0.200198 | 0.110357 | 444 | 444/444 | 444 | 1722 | 7 |
| ok | flat_slow03_clear03 | 5 | True | 8.026923 | -0.558737 | 0.565538 | 0.205208 | 470 | 470/471 | 470 | 1797 | 6 |
| ok | baseline | 6 | True | 8.112590 | -0.140302 | 0.199063 | 0.069632 | 0 | NA | 0 | 0 | 0 |
| ok | flat_noop | 6 | True | 8.128660 | -0.396173 | 0.401141 | 0.129324 | 454 | 454/455 | 454 | 1820 | 6 |
| ok | flat_slow03 | 6 | True | 8.079815 | -0.138621 | 0.317054 | 0.132806 | 458 | 458/458 | 458 | 1707 | 7 |
| ok | flat_clear03 | 6 | True | 8.117581 | -0.560700 | 0.563978 | 0.314734 | 438 | 438/439 | 438 | 1728 | 7 |
| ok | flat_slow03_clear03 | 6 | True | 8.185134 | 0.000666 | 0.192930 | 0.070614 | 451 | 451/451 | 451 | 1822 | 7 |

## Interpretation guide

- A candidate is promising only if it reaches all goals and improves both max_abs_y and mean_abs_y over baseline.
- Timeout rows are infrastructure failures, not locomotion failures.
- If no candidate beats baseline, keep active routing as scaffold/debug and move to learned/data-backed selector design.
