# TRACER Phase-K2B Corrected Segment Metrics Dataset Summary v0

- source summary: `datasets/phase_k/k1_long_baseline_vs_j19_profile_summary_v0.csv`
- output dataset: `datasets/phase_k/k2b_segment_metrics_from_k1_corrected_v0.csv`

K2B corrects the J7 context-level acceptance fields by using the already-corrected K1 summary columns.

## Mode/context segment summary

| mode | context | n | max_abs_y mean | mean_abs_y mean | abs_delta_y mean | j7 accepted mean | j7 projected mean |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | flat | 10 | 0.092109 | 0.040953 | 0.063815 | 0.00 | 0.00 |
| j19_profile | flat | 9 | 0.119429 | 0.060072 | 0.091060 | 448.22 | 448.22 |
| baseline | rough | 10 | 0.240604 | 0.172627 | 0.107659 | 0.00 | 0.00 |
| j19_profile | rough | 9 | 0.263205 | 0.202169 | 0.108396 | 0.00 | 0.00 |
| baseline | upslope | 10 | 0.154178 | 0.103558 | 0.084687 | 0.00 | 0.00 |
| j19_profile | upslope | 9 | 0.215489 | 0.144919 | 0.134281 | 0.00 | 0.00 |
| baseline | downslope | 10 | 0.303516 | 0.230116 | 0.126480 | 0.00 | 0.00 |
| j19_profile | downslope | 9 | 0.267163 | 0.186891 | 0.150738 | 0.00 | 0.00 |
| baseline | goal_flat | 10 | 0.296754 | 0.240172 | 0.090531 | 0.00 | 0.00 |
| j19_profile | goal_flat | 9 | 0.284505 | 0.239994 | 0.065890 | 0.00 | 0.00 |

## Delta summary: j19_profile - baseline

| context | Δ max_abs_y mean | Δ mean_abs_y mean | Δ abs_delta_y mean |
|---|---:|---:|---:|
| flat | +0.027321 | +0.019119 | +0.027245 |
| rough | +0.022600 | +0.029542 | +0.000737 |
| upslope | +0.061311 | +0.041361 | +0.049593 |
| downslope | -0.036353 | -0.043224 | +0.024258 |
| goal_flat | -0.012250 | -0.000178 | -0.024641 |

## Interpretation

- J19 profile has high flat acceptance, as expected.
- Segment metrics do not show clear flat improvement; flat, rough, and upslope are slightly worse on average.
- Downslope and goal_flat are slightly better on average.
- Current J19 should be described as a conservative active-routing scaffold, not as a robust performance-improving policy.
- Next step: K3 should build a conservative risk rule that keeps active routing only when historical segment metrics do not predict degradation.
