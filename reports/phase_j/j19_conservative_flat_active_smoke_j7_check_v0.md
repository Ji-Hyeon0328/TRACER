# TRACER Phase-J7 Guarded Ref Gate Log Check v0

- csv: `logs/phase_j/j19_conservative_flat_active_20260722_114201/j7/guarded_meta_action_ref_gate_j7_v0.csv`
- rows: `3252`
- accepted rows: `444`
- accept rate: `0.136531`
- first_t: `1784735002.350961`
- last_t: `1784735327.451149`

## Context counts

| context | rows |
|---|---:|
| downslope | 505 |
| flat | 445 |
| goal_flat | 1425 |
| rough | 446 |
| unknown | 7 |
| upslope | 424 |

## Decision reasons

| reason | rows |
|---|---:|
| stale_empirical_ref | 1078 |
| delta_vx_too_large | 932 |
| accepted_guarded | 444 |
| context_not_allowed_for_first_active | 426 |
| hold_or_goal_protected | 371 |
| goal_x_protected | 1 |

## Output source

| source | rows |
|---|---:|
| empirical | 1730 |
| failsafe_hold | 1078 |
| projected | 444 |

## Acceptance by context

| context | rows | accepted | accept rate |
|---|---:|---:|---:|
| downslope | 505 | 0 | 0.000000 |
| flat | 445 | 444 | 0.997753 |
| goal_flat | 1425 | 0 | 0.000000 |
| rough | 446 | 0 | 0.000000 |
| unknown | 7 | 0 | 0.000000 |
| upslope | 424 | 0 | 0.000000 |

## Acceptance by context/action

| context | action_id | rows | accepted | accept rate |
|---|---:|---:|---:|---:|
| downslope | 6 | 327 | 0 | 0.000000 |
| downslope | 7 | 154 | 0 | 0.000000 |
| downslope | 8 | 24 | 0 | 0.000000 |
| flat | 1 | 445 | 444 | 0.997753 |
| goal_flat | 8 | 1425 | 0 | 0.000000 |
| rough | 5 | 1 | 0 | 0.000000 |
| rough | 7 | 445 | 0 | 0.000000 |
| unknown | 7 | 7 | 0 | 0.000000 |
| upslope | 1 | 1 | 0 | 0.000000 |
| upslope | 5 | 423 | 0 | 0.000000 |

## Numeric summary: all vs accepted

| column | all mean | all min | all max | accepted mean | accepted min | accepted max |
|---|---:|---:|---:|---:|---:|---:|
| emp_vx | 0.127686 | 0.025000 | 0.210000 | 0.210000 | 0.210000 | 0.210000 |
| proj_vx | 0.106838 | 0.025000 | 0.220000 | 0.210000 | 0.210000 | 0.210000 |
| out_vx | 0.127686 | 0.025000 | 0.210000 | 0.210000 | 0.210000 | 0.210000 |
| delta_vx | -0.020848 | -0.177500 | 0.010000 | 0.000000 | 0.000000 | 0.000000 |
| emp_yaw_rate | 0.001810 | -0.001309 | 0.015473 | 0.000599 | -0.000771 | 0.001256 |
| proj_yaw_rate | 0.001258 | -0.000925 | 0.010476 | 0.000599 | -0.000771 | 0.001256 |
| out_yaw_rate | 0.001810 | -0.001309 | 0.015473 | 0.000599 | -0.000771 | 0.001256 |
| delta_yaw_rate | -0.000553 | -0.015473 | 0.000393 | 0.000000 | 0.000000 | 0.000000 |
| emp_body_h | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| proj_body_h | 0.317903 | 0.312000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| out_body_h | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| delta_body_h | -0.002097 | -0.008000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| emp_clearance | 0.046368 | 0.045000 | 0.055000 | 0.045000 | 0.045000 | 0.045000 |
| proj_clearance | 0.048784 | 0.042000 | 0.063000 | 0.045000 | 0.045000 | 0.045000 |
| out_clearance | 0.046368 | 0.045000 | 0.055000 | 0.045000 | 0.045000 | 0.045000 |
| delta_clearance | 0.002416 | -0.003000 | 0.008000 | 0.000000 | 0.000000 | 0.000000 |
| emp_age_s | 18.162159 | 0.072858 | 108.478880 | 0.078867 | 0.072858 | 0.081376 |
| proj_age_s | 0.008279 | 0.004987 | 0.011814 | 0.008196 | 0.005253 | 0.009955 |
| theta_age_s | 0.002979 | 0.000306 | 0.105256 | 0.002765 | 0.000391 | 0.105256 |

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
