# TRACER Phase-J2 Meta-Action Tabular Policy v0

This trains a simple tabular/context-guarded policy from J1 meta-action pseudo-labels.

- teacher csv: `datasets/phase_j/j1_meta_action_teacher_v0.csv`
- action bank: `configs/phase_j/meta_action_bank_v0.json`
- policy json: `models/phase_j/j2_meta_action_tabular_policy_v0.json`
- eval csv: `datasets/phase_j/j2_meta_action_tabular_policy_eval_v0.csv`
- total rows: `2124`
- high-confidence rows: `1686`
- rows used for policy table: `1686`
- margin threshold: `0.05`
- alpha: `0.05`

## Accuracy against J1 teacher

| split | rows | accuracy |
|---|---:|---:|
| all | 2124 | 0.749529 |
| high_conf | 1686 | 0.944247 |

## Margin summary

| split | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| all | 2124 | 0.228892 | 0.275641 | 0.018655 | 0.908063 |
| high_conf | 1686 | 0.277530 | 0.290247 | 0.063931 | 0.908063 |

## Context policy top actions

| context | rows used | top action | top action id |
|---|---:|---|---:|
| downslope | 497 | downslope_stable | 6 |
| goal_flat | 335 | goal_hold | 8 |
| rough | 434 | lateral_recovery_soft | 7 |
| upslope | 420 | upslope_push | 5 |

## Accuracy by context

| context | rows | correct | accuracy |
|---|---:|---:|---:|
| downslope | 497 | 469 | 0.943662 |
| flat | 437 | 0 | 0.000000 |
| goal_flat | 335 | 335 | 1.000000 |
| rough | 435 | 368 | 0.845977 |
| upslope | 420 | 420 | 1.000000 |

## Prediction reasons

| reason | rows |
|---|---:|
| context_top1 | 1246 |
| fallback | 437 |
| goal_context_guard | 335 |
| lateral_guard | 86 |
| goal_x_guard | 20 |

## Main confusions

| context | true | pred | rows |
|---|---|---|---:|
| flat | fast_motion | downslope_stable | 437 |
| rough | rough_stability | lateral_recovery_soft | 67 |
| downslope | lateral_recovery_soft | downslope_stable | 28 |

## Safe interpretation

- J2 is a supervised/tabular initialization from J1 pseudo-labels.
- J2 does not modify active control.
- This policy can be used as the first shadow meta-action selector.
- The next step should apply this policy in shadow mode to produce a_HL without changing `/tracer/mpc_reference`.
