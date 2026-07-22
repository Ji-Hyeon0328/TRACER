# TRACER Phase-J7 Guarded Ref Gate Log Check v0

- csv: `logs/phase_j/j7_for_j10c_override_shadow_20260721_233818/guarded_meta_action_ref_gate_j7_v0.csv`
- rows: `6141`
- accepted rows: `1408`
- accept rate: `0.229279`
- first_t: `1784691569.666032`
- last_t: `1784692183.666145`

## Context counts

| context | rows |
|---|---:|
| downslope | 532 |
| flat | 476 |
| goal_flat | 4262 |
| rough | 434 |
| unknown | 10 |
| upslope | 427 |

## Decision reasons

| reason | rows |
|---|---:|
| stale_empirical_ref | 3868 |
| accepted_guarded | 1408 |
| delta_vx_too_large | 446 |
| hold_or_goal_protected | 394 |
| goal_x_protected | 23 |
| missing_projected_ref | 1 |
| context_not_allowed_for_first_active | 1 |

## Output source

| source | rows |
|---|---:|
| failsafe_hold | 3869 |
| projected | 1408 |
| empirical | 864 |

## Acceptance by context

| context | rows | accepted | accept rate |
|---|---:|---:|---:|
| downslope | 532 | 507 | 0.953008 |
| flat | 476 | 474 | 0.995798 |
| goal_flat | 4262 | 0 | 0.000000 |
| rough | 434 | 0 | 0.000000 |
| unknown | 10 | 0 | 0.000000 |
| upslope | 427 | 427 | 1.000000 |

## Acceptance by context/action

| context | action_id | rows | accepted | accept rate |
|---|---:|---:|---:|---:|
| downslope | 6 | 531 | 507 | 0.954802 |
| downslope | 7 | 1 | 0 | 0.000000 |
| flat | 1 | 475 | 474 | 0.997895 |
| flat | 7 | 1 | 0 | 0.000000 |
| goal_flat | 8 | 4262 | 0 | 0.000000 |
| rough | 7 | 434 | 0 | 0.000000 |
| unknown | 7 | 10 | 0 | 0.000000 |
| upslope | 1 | 1 | 1 | 1.000000 |
| upslope | 5 | 426 | 426 | 1.000000 |

## Numeric summary: all vs accepted

| column | all mean | all min | all max | accepted mean | accepted min | accepted max |
|---|---:|---:|---:|---:|---:|---:|
| emp_vx | 0.080921 | 0.025000 | 0.210000 | 0.207299 | 0.202500 | 0.210000 |
| proj_vx | 0.072914 | 0.025000 | 0.220000 | 0.199394 | 0.178200 | 0.220000 |
| out_vx | 0.079078 | 0.025000 | 0.220000 | 0.199394 | 0.178200 | 0.220000 |
| delta_vx | -0.007986 | -0.177500 | 0.010000 | -0.007905 | -0.024300 | 0.010000 |
| emp_yaw_rate | 0.001482 | -0.001615 | 0.013738 | 0.004366 | -0.001615 | 0.013738 |
| proj_yaw_rate | 0.001184 | -0.001615 | 0.012365 | 0.003915 | -0.001615 | 0.012365 |
| out_yaw_rate | 0.001379 | -0.001615 | 0.013666 | 0.003915 | -0.001615 | 0.012365 |
| delta_yaw_rate | -0.000299 | -0.013666 | 0.000161 | -0.000451 | -0.001415 | 0.000161 |
| emp_body_h | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| proj_body_h | 0.319073 | 0.312000 | 0.320000 | 0.318560 | 0.316000 | 0.320000 |
| out_body_h | 0.319670 | 0.316000 | 0.320000 | 0.318560 | 0.316000 | 0.320000 |
| delta_body_h | -0.000927 | -0.008000 | 0.000000 | -0.001440 | -0.004000 | 0.000000 |
| emp_clearance | 0.045708 | 0.045000 | 0.055000 | 0.045000 | 0.045000 | 0.045000 |
| proj_clearance | 0.046680 | 0.042000 | 0.063000 | 0.046634 | 0.042000 | 0.049000 |
| out_clearance | 0.046083 | 0.042000 | 0.055000 | 0.046634 | 0.042000 | 0.049000 |
| delta_clearance | 0.000972 | -0.006000 | 0.018000 | 0.001634 | -0.003000 | 0.004000 |
| emp_age_s | 122.294636 | 0.063762 | 387.469029 | 0.069026 | 0.063762 | 0.071758 |
| proj_age_s | 0.083397 | 0.078069 | 1.082636 | 0.082638 | 0.079190 | 0.085260 |
| theta_age_s | 0.035853 | 0.025000 | 1.633489 | 0.033610 | 0.025000 | 0.036756 |

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
