# TRACER Phase-J3 Meta-Action Shadow Log Check v0

- csv: `logs/phase_j/j3_meta_action_shadow_active_20260721_174204/meta_action_shadow_j3_v0.csv`
- rows: `6893`
- first_t: `1784670125.133655`
- last_t: `1784670814.333488`

## Context counts

| context | rows |
|---|---:|
| downslope | 499 |
| flat | 488 |
| goal_flat | 4365 |
| rough | 416 |
| unknown | 695 |
| upslope | 430 |

## Action distribution

| action | rows |
|---|---:|
| goal_hold | 4393 |
| downslope_stable | 1166 |
| fast_motion | 488 |
| upslope_push | 430 |
| lateral_recovery_soft | 416 |

## Action by context

| context | action | rows |
|---|---|---:|
| downslope | downslope_stable | 471 |
| downslope | goal_hold | 28 |
| flat | fast_motion | 488 |
| goal_flat | goal_hold | 4365 |
| rough | lateral_recovery_soft | 416 |
| unknown | downslope_stable | 695 |
| upslope | upslope_push | 430 |

## Selection reasons

| reason | rows |
|---|---:|
| goal_context_guard | 4365 |
| context_top1 | 1805 |
| fallback | 695 |
| goal_x_guard | 28 |

## Numeric summary

| column | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| x | 6198 | 6.889705 | 2.288889 | -0.002458 | 8.113326 |
| y | 6198 | -0.161138 | 0.065401 | -0.375944 | 0.042201 |
| beta_motion | 6164 | 0.238571 | 0.097111 | 0.181687 | 0.523487 |
| beta_stability | 6164 | 0.511467 | 0.098253 | 0.198042 | 0.562934 |
| beta_energy | 6164 | 0.249962 | 0.029581 | 0.166746 | 0.323973 |
| ram_slip | 6170 | 0.033955 | 0.072674 | 0.000000 | 0.250000 |
| ram_rough | 6170 | 0.092099 | 0.094051 | 0.050000 | 0.400000 |
| ram_sigma | 6170 | 0.100891 | 0.100481 | 0.050000 | 0.350000 |
| vx_scale | 6893 | 0.307518 | 0.417031 | 0.000000 | 1.080000 |
| body_h_delta | 6893 | -0.001498 | 0.002776 | -0.008000 | 0.000000 |
| clearance_delta | 6893 | 0.001197 | 0.002602 | -0.003000 | 0.008000 |
| stability_bias | 6893 | 0.231605 | 0.102053 | -0.100000 | 0.350000 |
| energy_bias | 6893 | 0.031104 | 0.093745 | -0.150000 | 0.100000 |

## Quick interpretation

- J3 is shadow-only and should not modify `/tracer/mpc_reference`.
- Expected mapping: flat->fast_motion, upslope->upslope_push, downslope->downslope_stable, rough->lateral_recovery_soft/rough_stability, goal_flat->goal_hold.
- If context/action transitions are visible, the meta-action policy runtime path is connected.
