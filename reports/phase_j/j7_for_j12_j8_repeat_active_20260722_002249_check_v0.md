# TRACER Phase-J7 Guarded Ref Gate Log Check v0

- csv: `logs/phase_j/j7_for_j12_j8_repeat_active_20260722_002249/guarded_meta_action_ref_gate_j7_v0.csv`
- rows: `3556`
- accepted rows: `875`
- accept rate: `0.246063`
- first_t: `1784694247.596837`
- last_t: `1784694603.097216`

## Context counts

| context | rows |
|---|---:|
| downslope | 504 |
| flat | 426 |
| goal_flat | 1736 |
| rough | 439 |
| unknown | 1 |
| upslope | 450 |

## Decision reasons

| reason | rows |
|---|---:|
| stale_empirical_ref | 1390 |
| delta_vx_too_large | 916 |
| accepted_guarded | 875 |
| hold_or_goal_protected | 372 |
| missing_projected_ref | 1 |
| context_not_allowed_for_first_active | 1 |
| goal_x_protected | 1 |

## Output source

| source | rows |
|---|---:|
| failsafe_hold | 1391 |
| empirical | 1290 |
| projected | 875 |

## Acceptance by context

| context | rows | accepted | accept rate |
|---|---:|---:|---:|
| downslope | 504 | 0 | 0.000000 |
| flat | 426 | 425 | 0.997653 |
| goal_flat | 1736 | 0 | 0.000000 |
| rough | 439 | 0 | 0.000000 |
| unknown | 1 | 0 | 0.000000 |
| upslope | 450 | 450 | 1.000000 |

## Acceptance by context/action

| context | action_id | rows | accepted | accept rate |
|---|---:|---:|---:|---:|
| downslope | 7 | 478 | 0 | 0.000000 |
| downslope | 8 | 26 | 0 | 0.000000 |
| flat | 1 | 426 | 425 | 0.997653 |
| goal_flat | 8 | 1736 | 0 | 0.000000 |
| rough | 7 | 439 | 0 | 0.000000 |
| unknown | 7 | 1 | 0 | 0.000000 |
| upslope | 1 | 1 | 1 | 1.000000 |
| upslope | 5 | 449 | 449 | 1.000000 |

## Numeric summary: all vs accepted

| column | all mean | all min | all max | accepted mean | accepted min | accepted max |
|---|---:|---:|---:|---:|---:|---:|
| emp_vx | 0.118655 | 0.025000 | 0.210000 | 0.210000 | 0.210000 | 0.210000 |
| proj_vx | 0.098433 | 0.025000 | 0.220000 | 0.210579 | 0.201600 | 0.220000 |
| out_vx | 0.118746 | 0.025000 | 0.220000 | 0.210579 | 0.201600 | 0.220000 |
| delta_vx | -0.020196 | -0.177500 | 0.010000 | 0.000579 | -0.008400 | 0.010000 |
| emp_yaw_rate | 0.005539 | -0.000921 | 0.021059 | 0.003922 | -0.000921 | 0.013225 |
| proj_yaw_rate | 0.004025 | -0.000921 | 0.014741 | 0.003731 | -0.000921 | 0.012563 |
| out_yaw_rate | 0.005492 | -0.000921 | 0.021059 | 0.003731 | -0.000921 | 0.012563 |
| delta_yaw_rate | -0.001516 | -0.013475 | 0.000147 | -0.000191 | -0.000896 | 0.000147 |
| emp_body_h | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| proj_body_h | 0.317934 | 0.312000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| out_body_h | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| delta_body_h | -0.002066 | -0.008000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| emp_clearance | 0.046232 | 0.045000 | 0.055000 | 0.045000 | 0.045000 | 0.045000 |
| proj_clearance | 0.048443 | 0.042000 | 0.063000 | 0.045584 | 0.042000 | 0.049000 |
| out_clearance | 0.046375 | 0.042000 | 0.055000 | 0.045584 | 0.042000 | 0.049000 |
| delta_clearance | 0.002211 | -0.003000 | 0.018000 | 0.000584 | -0.003000 | 0.004000 |
| emp_age_s | 27.502309 | 0.038669 | 139.741841 | 0.041667 | 0.038742 | 0.043258 |
| proj_age_s | 0.077508 | 0.069539 | 1.574575 | 0.074197 | 0.069539 | 0.075799 |
| theta_age_s | 0.037082 | 0.025730 | 2.131136 | 0.030609 | 0.025730 | 0.032688 |

## Output safety violations

| check | count |
|---|---:|
| out_vx_low | 0 |
| out_vx_high | 0 |
| out_yaw_low | 0 |
| out_yaw_high | 0 |
| out_body_h_low | 0 |
| out_body_h_high | 0 |
| out_clearance_low | 0 |
| out_clearance_high | 0 |

## Safe interpretation

- J7 is still shadow-only if output topic is `/tracer/meta_action_guarded_ref_shadow`.
- Expected first active candidates should be accepted mainly in flat and upslope.
- Rejected rows should fall back to empirical references.
- Do not route this node to `/tracer/mpc_reference` until the shadow gate report is checked.
