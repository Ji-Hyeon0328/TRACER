# TRACER Phase-J20 J19-Profile Repeat Partial Summary v0

- log root: `logs/phase_j/j20_j19_profile_repeat_20260722_115530`
- summary csv: `logs/phase_j/j20_j19_profile_repeat_20260722_115530/j20_j19_profile_repeat_summary_v0.csv`
- status: partial; run was stopped because idx11 j19_profile appeared stuck during reset/y-offset stage.

## Group summary

| mode | n | goals | max_abs_y mean | max_abs_y min | max_abs_y max | mean_abs_y mean |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 11 | 11 | 0.329457 | 0.185757 | 0.471330 | 0.151440 |
| j19_profile | 10 | 10 | 0.356326 | 0.106567 | 0.559871 | 0.178996 |

## Per-run results

| mode | idx | goal | final_x | final_y | max_abs_y | mean_abs_y | j7 accepted | flat acc | projected | empirical | failsafe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | True | 8.140051 | -0.031465 | 0.185757 | 0.093794 | 0 | NA | 0 | 0 | 0 |
| j19_profile | 1 | True | 8.057371 | -0.088826 | 0.300793 | 0.140311 | 0 | NA | 0 | 0 | 0 |
| baseline | 2 | True | 8.129659 | 0.059193 | 0.414844 | 0.154725 | 0 | NA | 0 | 0 | 0 |
| j19_profile | 2 | True | 8.022588 | -0.412955 | 0.559871 | 0.248715 | 0 | NA | 0 | 0 | 0 |
| baseline | 3 | True | 8.126185 | -0.274655 | 0.463595 | 0.203585 | 0 | NA | 0 | 0 | 0 |
| j19_profile | 3 | True | 7.992146 | 0.123611 | 0.359983 | 0.149408 | 0 | NA | 0 | 0 | 0 |
| baseline | 4 | True | 8.056612 | -0.184927 | 0.194159 | 0.085834 | 0 | NA | 0 | 0 | 0 |
| j19_profile | 4 | True | 8.018371 | -0.125376 | 0.351122 | 0.217969 | 0 | NA | 0 | 0 | 0 |
| baseline | 5 | True | 7.973979 | -0.212529 | 0.385933 | 0.229182 | 0 | NA | 0 | 0 | 0 |
| j19_profile | 5 | True | 8.036326 | 0.059396 | 0.106567 | 0.037551 | 0 | NA | 0 | 0 | 0 |
| baseline | 6 | True | 8.137876 | -0.459827 | 0.471330 | 0.227908 | 0 | NA | 0 | 0 | 0 |
| j19_profile | 6 | True | 8.087878 | -0.313736 | 0.364855 | 0.192592 | 0 | NA | 0 | 0 | 0 |
| baseline | 7 | True | 8.080694 | -0.211735 | 0.237229 | 0.080437 | 0 | NA | 0 | 0 | 0 |
| j19_profile | 7 | True | 8.147857 | -0.553703 | 0.555640 | 0.309042 | 0 | NA | 0 | 0 | 0 |
| baseline | 8 | True | 8.183367 | -0.268904 | 0.268904 | 0.086535 | 0 | NA | 0 | 0 | 0 |
| j19_profile | 8 | True | 8.120835 | 0.443565 | 0.484769 | 0.259571 | 0 | NA | 0 | 0 | 0 |
| baseline | 9 | True | 8.163114 | 0.304097 | 0.426616 | 0.243404 | 0 | NA | 0 | 0 | 0 |
| j19_profile | 9 | True | 8.030662 | 0.002597 | 0.137079 | 0.047657 | 0 | NA | 0 | 0 | 0 |
| baseline | 10 | True | 8.095024 | 0.081062 | 0.194745 | 0.063248 | 0 | NA | 0 | 0 | 0 |
| j19_profile | 10 | True | 8.204328 | 0.340139 | 0.342585 | 0.187140 | 0 | NA | 0 | 0 | 0 |
| baseline | 11 | True | 8.194246 | -0.265776 | 0.380912 | 0.197184 | 0 | NA | 0 | 0 | 0 |

## Interpretation

- This is a partial repeat study, not a completed J20 N=12 result.
- Existing completed rows are still useful for comparing baseline and J19 profile trends.
- The stopped run should not be counted as a mission failure; it is an infrastructure/reset hang.
- If the partial trend is clear enough, use this as an audit result. Otherwise rerun J20 with smaller batches or a reset timeout guard.
