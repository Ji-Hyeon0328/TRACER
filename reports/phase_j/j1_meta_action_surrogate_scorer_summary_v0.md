# TRACER Phase-J1 Meta-Action Surrogate Scorer v0

This scores the J0 discrete meta-action bank using D7 beta, context, RAM proxy, and drift-style runtime features.

- D7 csv: `logs/phase_d5_shadow_20260721_164242_trial_1/d7_objective_selector_shadow_v0.csv`
- action bank: `configs/phase_j/meta_action_bank_v0.json`
- scores csv: `datasets/phase_j/j1_meta_action_surrogate_scores_v0.csv`
- teacher csv: `datasets/phase_j/j1_meta_action_teacher_v0.csv`
- scorer json: `models/phase_j/j1_meta_action_surrogate_scorer_v0.json`
- D7 rows used: `2124`
- score rows: `19116`
- actions: `9`

## Beta source columns

| beta columns | rows |
|---|---:|
| actual_beta_motion,actual_beta_stability,actual_beta_energy | 2124 |

## Teacher action distribution

| action | rows |
|---|---:|
| downslope_stable | 449 |
| fast_motion | 437 |
| upslope_push | 420 |
| lateral_recovery_soft | 396 |
| goal_hold | 355 |
| rough_stability | 67 |

## Teacher action by context

| context | action | rows |
|---|---|---:|
| downslope | downslope_stable | 449 |
| downslope | lateral_recovery_soft | 28 |
| downslope | goal_hold | 20 |
| flat | fast_motion | 437 |
| goal_flat | goal_hold | 335 |
| rough | lateral_recovery_soft | 368 |
| rough | rough_stability | 67 |
| upslope | upslope_push | 420 |

## Top-2 margin summary

| n | mean | std | min | max |
|---:|---:|---:|---:|---:|
| 2124 | 0.228892 | 0.275641 | 0.018655 | 0.908063 |

## Safe interpretation

- J1 produces heuristic pseudo-labels for meta-action selection.
- J1 does not modify active control.
- The labels should be treated as contextual-bandit / offline-RL initialization, not final RL.
- If one action dominates too strongly, J2 should add entropy/balancing before training a policy.
