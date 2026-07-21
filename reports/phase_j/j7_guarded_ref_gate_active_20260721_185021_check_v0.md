# TRACER Phase-J7 Guarded Ref Gate Log Check v0

- csv: `logs/phase_j/j7_guarded_ref_gate_active_20260721_185021/guarded_meta_action_ref_gate_j7_v0.csv`
- rows: `2412`
- accepted rows: `873`
- accept rate: `0.361940`
- first_t: `1784674314.905928`
- last_t: `1784674556.006220`

## Context counts

| context | rows |
|---|---:|
| downslope | 484 |
| flat | 420 |
| goal_flat | 590 |
| rough | 453 |
| unknown | 11 |
| upslope | 454 |

## Decision reasons

| reason | rows |
|---|---:|
| delta_vx_too_large | 927 |
| accepted_guarded | 873 |
| hold_or_goal_protected | 369 |
| stale_empirical_ref | 239 |
| context_not_allowed_for_first_active | 2 |
| missing_projected_ref | 1 |
| goal_x_protected | 1 |

## Output source

| source | rows |
|---|---:|
| empirical | 1299 |
| projected | 873 |
| failsafe_hold | 240 |

## Acceptance by context

| context | rows | accepted | accept rate |
|---|---:|---:|---:|
| downslope | 484 | 0 | 0.000000 |
| flat | 420 | 419 | 0.997619 |
| goal_flat | 590 | 0 | 0.000000 |
| rough | 453 | 0 | 0.000000 |
| unknown | 11 | 0 | 0.000000 |
| upslope | 454 | 454 | 1.000000 |

## Acceptance by context/action

| context | action_id | rows | accepted | accept rate |
|---|---:|---:|---:|---:|
| downslope | 6 | 466 | 0 | 0.000000 |
| downslope | 8 | 18 | 0 | 0.000000 |
| flat | 1 | 420 | 419 | 0.997619 |
| goal_flat | 8 | 590 | 0 | 0.000000 |
| rough | 5 | 1 | 0 | 0.000000 |
| rough | 7 | 452 | 0 | 0.000000 |
| unknown | 7 | 11 | 0 | 0.000000 |
| upslope | 5 | 454 | 454 | 1.000000 |

## Numeric summary: all vs accepted

| column | all mean | all min | all max | accepted mean | accepted min | accepted max |
|---|---:|---:|---:|---:|---:|---:|
| emp_vx | 0.163041 | 0.025000 | 0.210000 | 0.210000 | 0.210000 | 0.210000 |
| proj_vx | 0.138741 | 0.025000 | 0.220000 | 0.210452 | 0.201600 | 0.220000 |
| out_vx | 0.163128 | 0.025000 | 0.220000 | 0.210452 | 0.201600 | 0.220000 |
| delta_vx | -0.024280 | -0.177500 | 0.010000 | 0.000452 | -0.008400 | 0.010000 |
| emp_yaw_rate | 0.001250 | -0.002026 | 0.005709 | 0.000271 | -0.002026 | 0.002242 |
| proj_yaw_rate | 0.000928 | -0.002026 | 0.004279 | 0.000229 | -0.002026 | 0.002129 |
| out_yaw_rate | 0.001235 | -0.002026 | 0.005709 | 0.000229 | -0.002026 | 0.002129 |
| delta_yaw_rate | -0.000323 | -0.003383 | 0.000146 | -0.000042 | -0.000895 | 0.000146 |
| emp_body_h | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| proj_body_h | 0.317304 | 0.312000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| out_body_h | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| delta_body_h | -0.002696 | -0.008000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| emp_clearance | 0.046874 | 0.045000 | 0.055000 | 0.045000 | 0.045000 | 0.045000 |
| proj_clearance | 0.049416 | 0.042000 | 0.063000 | 0.045632 | 0.042000 | 0.049000 |
| out_clearance | 0.047103 | 0.042000 | 0.055000 | 0.045632 | 0.042000 | 0.049000 |
| delta_clearance | 0.002542 | -0.006000 | 0.018000 | 0.000632 | -0.003000 | 0.004000 |
| emp_age_s | 1.267341 | 0.002176 | 24.607608 | 0.007849 | 0.002176 | 0.009660 |
| proj_age_s | 0.034575 | 0.029019 | 0.933182 | 0.032731 | 0.029735 | 0.035055 |
| theta_age_s | 0.035650 | 0.025927 | 1.729444 | 0.029357 | 0.026253 | 0.031463 |

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
