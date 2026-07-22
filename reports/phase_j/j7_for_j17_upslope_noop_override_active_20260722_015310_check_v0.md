# TRACER Phase-J7 Guarded Ref Gate Log Check v0

- csv: `logs/phase_j/j7_for_j17_upslope_noop_override_active_20260722_015310/guarded_meta_action_ref_gate_j7_v0.csv`
- rows: `2532`
- accepted rows: `431`
- accept rate: `0.170221`
- first_t: `1784699660.970849`
- last_t: `1784699914.070905`

## Context counts

| context | rows |
|---|---:|
| downslope | 518 |
| flat | 443 |
| goal_flat | 700 |
| rough | 433 |
| unknown | 7 |
| upslope | 431 |

## Decision reasons

| reason | rows |
|---|---:|
| delta_vx_too_large | 937 |
| context_not_allowed_for_first_active | 443 |
| accepted_guarded | 431 |
| stale_empirical_ref | 363 |
| hold_or_goal_protected | 357 |
| goal_x_protected | 1 |

## Output source

| source | rows |
|---|---:|
| empirical | 1738 |
| projected | 431 |
| failsafe_hold | 363 |

## Acceptance by context

| context | rows | accepted | accept rate |
|---|---:|---:|---:|
| downslope | 518 | 0 | 0.000000 |
| flat | 443 | 0 | 0.000000 |
| goal_flat | 700 | 0 | 0.000000 |
| rough | 433 | 0 | 0.000000 |
| unknown | 7 | 0 | 0.000000 |
| upslope | 431 | 431 | 1.000000 |

## Acceptance by context/action

| context | action_id | rows | accepted | accept rate |
|---|---:|---:|---:|---:|
| downslope | 6 | 498 | 0 | 0.000000 |
| downslope | 8 | 20 | 0 | 0.000000 |
| flat | 1 | 442 | 0 | 0.000000 |
| flat | 7 | 1 | 0 | 0.000000 |
| goal_flat | 8 | 700 | 0 | 0.000000 |
| rough | 5 | 1 | 0 | 0.000000 |
| rough | 7 | 432 | 0 | 0.000000 |
| unknown | 7 | 7 | 0 | 0.000000 |
| upslope | 1 | 1 | 1 | 1.000000 |
| upslope | 5 | 430 | 430 | 1.000000 |

## Numeric summary: all vs accepted

| column | all mean | all min | all max | accepted mean | accepted min | accepted max |
|---|---:|---:|---:|---:|---:|---:|
| emp_vx | 0.157308 | 0.025000 | 0.210000 | 0.210000 | 0.210000 | 0.210000 |
| proj_vx | 0.135526 | 0.025000 | 0.220000 | 0.210023 | 0.210000 | 0.220000 |
| out_vx | 0.157312 | 0.025000 | 0.220000 | 0.210023 | 0.210000 | 0.220000 |
| delta_vx | -0.021782 | -0.177500 | 0.010000 | 0.000023 | 0.000000 | 0.010000 |
| emp_yaw_rate | 0.002038 | -0.008220 | 0.011079 | 0.005902 | 0.002138 | 0.009372 |
| proj_yaw_rate | 0.001761 | -0.005929 | 0.009421 | 0.005902 | 0.002138 | 0.009372 |
| out_yaw_rate | 0.002038 | -0.008220 | 0.011079 | 0.005902 | 0.002138 | 0.009372 |
| delta_yaw_rate | -0.000277 | -0.003324 | 0.008220 | 0.000000 | 0.000000 | 0.000000 |
| emp_body_h | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| proj_body_h | 0.317430 | 0.312000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| out_body_h | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| delta_body_h | -0.002570 | -0.008000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| emp_clearance | 0.046706 | 0.045000 | 0.055000 | 0.045000 | 0.045000 | 0.045000 |
| proj_clearance | 0.048358 | 0.042000 | 0.063000 | 0.044993 | 0.042000 | 0.045000 |
| out_clearance | 0.046705 | 0.042000 | 0.055000 | 0.044993 | 0.042000 | 0.045000 |
| delta_clearance | 0.001652 | -0.003000 | 0.008000 | -0.000007 | -0.003000 | 0.000000 |
| emp_age_s | 2.758828 | 0.059732 | 36.962776 | 0.062841 | 0.060598 | 0.063916 |
| proj_age_s | 0.050222 | 0.046483 | 0.548755 | 0.049739 | 0.047508 | 0.051327 |
| theta_age_s | 0.081488 | 0.075836 | 0.979634 | 0.079777 | 0.076556 | 0.081472 |

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
