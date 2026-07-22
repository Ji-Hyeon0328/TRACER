# TRACER Phase-J18 Repeat Study Corrected Summary v0

- source csv: `logs/phase_j/j18_repeat_study_20260722_020336/j18_repeat_study_summary_v0.csv`

This corrected report uses the mission metrics from the J18 summary CSV and re-parses generated J7 checker markdown reports for active-mode acceptance statistics.

## Mission group summary

| mode | n | goals | max_abs_y mean | max_abs_y min | max_abs_y max | mean_abs_y mean |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 6 | 6 | 0.369372 | 0.203091 | 0.534749 | 0.164393 |
| flat_noop | 6 | 6 | 0.256871 | 0.184639 | 0.441479 | 0.111159 |
| upslope_noop | 6 | 5 | 0.611617 | 0.191963 | 2.004607 | 0.204426 |

## Per-run corrected J7 summary

| mode | idx | goal | max_abs_y | mean_abs_y | j7 accepted | j7 accept rate | projected | empirical | failsafe | flat acc | upslope acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | True | 0.534749 | 0.235157 | NA | NA | NA | NA | NA | NA | NA |
| flat_noop | 1 | True | 0.259328 | 0.139709 | 456 | 0.209752 | 456 | 1711 | NA | 456/458 | 0/433 |
| upslope_noop | 1 | True | 0.439779 | 0.162476 | 445 | 0.195949 | 445 | 1819 | NA | 0/453 | 445/445 |
| baseline | 2 | True | 0.501532 | 0.260088 | NA | NA | NA | NA | NA | NA | NA |
| flat_noop | 2 | True | 0.230957 | 0.087657 | 445 | 0.195175 | 445 | 1827 | NA | 445/446 | 0/419 |
| upslope_noop | 2 | True | 0.390388 | 0.197955 | 413 | 0.189972 | 413 | 1754 | NA | 0/449 | 413/413 |
| baseline | 3 | True | 0.226340 | 0.100554 | NA | NA | NA | NA | NA | NA | NA |
| flat_noop | 3 | True | 0.441479 | 0.200978 | 446 | 0.204306 | 446 | 1730 | NA | 446/447 | 0/416 |
| upslope_noop | 3 | True | 0.203747 | 0.088702 | 413 | 0.189885 | 413 | 1754 | NA | 0/451 | 413/413 |
| baseline | 4 | True | 0.307759 | 0.075052 | NA | NA | NA | NA | NA | NA | NA |
| flat_noop | 4 | True | 0.234830 | 0.086030 | 445 | 0.204692 | 445 | 1723 | NA | 445/445 | 0/443 |
| upslope_noop | 4 | False | 2.004607 | 0.520830 | 416 | 0.222579 | 416 | 1446 | NA | 0/439 | 416/416 |
| baseline | 5 | True | 0.203091 | 0.067516 | NA | NA | NA | NA | NA | NA | NA |
| flat_noop | 5 | True | 0.189990 | 0.076314 | 450 | 0.206992 | 450 | 1718 | NA | 450/450 | 0/421 |
| upslope_noop | 5 | True | 0.439220 | 0.190880 | 434 | 0.199632 | 434 | 1734 | NA | 0/448 | 434/434 |
| baseline | 6 | True | 0.442763 | 0.247991 | NA | NA | NA | NA | NA | NA | NA |
| flat_noop | 6 | True | 0.184639 | 0.076268 | 447 | 0.205801 | 447 | 1719 | NA | 447/448 | 0/422 |
| upslope_noop | 6 | True | 0.191963 | 0.065714 | 439 | 0.193222 | 439 | 1826 | NA | 0/437 | 439/439 |

## Interpretation

- Baseline variance is non-negligible, but all baseline runs reached the goal.
- Flat no-op remains the most promising active route candidate if the corrected J7 reports confirm high flat acceptance.
- Upslope no-op should remain frozen because it has a catastrophic failure run and worse group-level max_abs_y.
- Original flat fast_motion and upslope_push action values should not be promoted.
