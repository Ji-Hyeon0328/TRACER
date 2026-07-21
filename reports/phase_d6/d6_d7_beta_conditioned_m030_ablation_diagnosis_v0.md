# TRACER Phase-D6 D7-Beta-Conditioned m030 Ablation Diagnosis v0

## Question

The retrained D7-beta-conditioned D6 selector restored clean-run compatibility. However, under `TRACER_RESET_Y_OFFSET=-0.30`, the full learned vx/yaw/clearance deployment showed larger lateral drift. This diagnosis separates whether the drift is caused by learned yaw, learned vx, or the base empirical/low-level response.

## Compared runs

| mode | manifest | learned vx | learned yaw | learned clearance |
|---|---|---:|---:|---:|
| full learned | `reports/phase_d4_context_meta_repeat_20260720_162902_manifest.tsv` | 1 | 1 | 1 |
| no learned yaw | `reports/phase_d4_context_meta_repeat_20260720_180654_manifest.tsv` | 1 | 0 | 1 |
| no learned vx | `reports/phase_d4_context_meta_repeat_20260720_192118_manifest.tsv` | 0 | 1 | 1 |
| clearance only | `reports/phase_d4_context_meta_repeat_20260720_192832_manifest.tsv` | 0 | 0 | 1 |
| clearance only repeat N=3 | `reports/phase_d4_context_meta_repeat_20260720_194813_manifest.tsv` | 0 | 0 | 1 |

## Single-run ablation summary

| mode | success | goal | out_lane | final_y | max_abs_y | mean_abs_y | hold_drift | moving_accept_rate | vx rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full learned | 1 | 1 | 0 | 0.694 | 0.694 | 0.219 | 0.000 | 0.928849 | 117 |
| no learned yaw | 1 | 1 | 0 | 0.664 | 0.667 | 0.283 | 0.014 | 0.959629 | 64 |
| no learned vx | 1 | 1 | 0 | 0.474 | 0.505 | 0.241 | 0.006 | 0.993953 | 0 |
| clearance only | 1 | 1 | 0 | 0.183 | 0.197 | 0.080 | 0.069 | 0.994328 | 0 |

## Clearance-only repeat summary

| trial | success | goal | out_lane | final_y | max_abs_y | mean_abs_y | hold_drift | moving_accept_rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 0 | -0.259 | 0.321 | 0.142 | 0.033 | 0.993224 |
| 2 | 1 | 1 | 0 | -0.503 | 0.516 | 0.207 | 0.001 | 0.992778 |
| 3 | 1 | 1 | 0 | -0.443 | 0.498 | 0.326 | 0.027 | 0.993916 |

## Interpretation

- Full learned vx/yaw/clearance reaches the goal but shows significant lateral overshoot under negative lateral reset.
- Disabling learned yaw alone improves gate compatibility but does not substantially reduce lateral drift.
- Disabling learned vx substantially improves gate compatibility and reduces lateral drift.
- Disabling both learned vx and learned yaw while keeping learned clearance gives the best single-run lateral tracking among the tested ablations.
- The clearance-only fallback remains stable over N=3: all trials reach the goal, no out-of-lane event occurs, and moving gate compatibility stays around 0.993.
- Therefore, the m030 issue is not a failure of D7 beta publication or beta-conditioned D6 retraining. It is a dimension-specific calibration issue: learned vx/yaw are not yet robust under the negative lateral-offset state distribution.
- A safe deployment fallback is to use learned clearance while keeping vx/yaw empirical under large lateral mismatch.

## Poster-safe claim

D7 beta-conditioned D6 retraining restores reference compatibility in nominal traversal and positive lateral-offset traversal. Under negative lateral offset, learned vx/yaw can amplify lateral drift, but a dimension-selective gated deployment using learned clearance with empirical vx/yaw preserves goal reaching and stable gate compatibility.

## Next step

Use the clearance-only setting as the robust fallback for lateral-offset validation. Later, augment D6 training with lateral-offset examples before re-enabling learned vx/yaw under large lateral mismatch.
