# TRACER Phase-K2 Segment Metrics Dataset Summary v0

- source summary: `datasets/phase_k/k1_long_baseline_vs_j19_profile_summary_v0.csv`
- output dataset: `datasets/phase_k/k2_segment_metrics_from_k1_v0.csv`

## Mode/context segment summary

| mode | context | n | max_abs_y mean | mean_abs_y mean | abs_delta_y mean | j7 accepted mean | j7 projected mean |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | flat | 10 | 0.092109 | 0.040953 | 0.063815 | 0.00 | 0.00 |
| j19_profile | flat | 9 | 0.119429 | 0.060072 | 0.091060 | 0.00 | 0.00 |
| baseline | rough | 10 | 0.240604 | 0.172627 | 0.107659 | 0.00 | 0.00 |
| j19_profile | rough | 9 | 0.263205 | 0.202169 | 0.108396 | 0.00 | 0.00 |
| baseline | upslope | 10 | 0.154178 | 0.103558 | 0.084687 | 0.00 | 0.00 |
| j19_profile | upslope | 9 | 0.215489 | 0.144919 | 0.134281 | 0.00 | 0.00 |
| baseline | downslope | 10 | 0.303516 | 0.230116 | 0.126480 | 0.00 | 0.00 |
| j19_profile | downslope | 9 | 0.267163 | 0.186891 | 0.150738 | 0.00 | 0.00 |
| baseline | goal_flat | 10 | 0.296754 | 0.240172 | 0.090531 | 0.00 | 0.00 |
| j19_profile | goal_flat | 9 | 0.284505 | 0.239994 | 0.065890 | 0.00 | 0.00 |

## Interpretation guide

- If j19_profile improves flat but worsens later contexts, the active flat command may perturb transition state.
- If j19_profile and baseline are similar in every non-flat context, J19 can be kept as a neutral safe-routing scaffold.
- If flat is improved and non-flat is not worse, J19 flat conservative routing becomes a stronger safe candidate.
- This dataset is intended for Phase-K3 conservative action selector / risk analysis.
