# TRACER Phase-J7 Guarded Ref Gate Log Check v0

- csv: `logs/phase_j/j7_for_j10b_downslope_force6_shadow_20260721_222853/guarded_meta_action_ref_gate_j7_v0.csv`
- rows: `2348`
- accepted rows: `910`
- accept rate: `0.387564`
- first_t: `1784687401.984415`
- last_t: `1784687636.684711`

## Context counts

| context | rows |
|---|---:|
| downslope | 510 |
| flat | 482 |
| goal_flat | 491 |
| rough | 428 |
| unknown | 8 |
| upslope | 429 |

## Decision reasons

| reason | rows |
|---|---:|
| stale_theta | 918 |
| accepted_guarded | 910 |
| delta_vx_too_large | 442 |
| stale_empirical_ref | 76 |
| missing_projected_ref | 1 |
| context_not_allowed_for_first_active | 1 |

## Output source

| source | rows |
|---|---:|
| failsafe_hold | 995 |
| projected | 910 |
| empirical | 443 |

## Acceptance by context

| context | rows | accepted | accept rate |
|---|---:|---:|---:|
| downslope | 510 | 0 | 0.000000 |
| flat | 482 | 481 | 0.997925 |
| goal_flat | 491 | 0 | 0.000000 |
| rough | 428 | 0 | 0.000000 |
| unknown | 8 | 0 | 0.000000 |
| upslope | 429 | 429 | 1.000000 |

## Acceptance by context/action

| context | action_id | rows | accepted | accept rate |
|---|---:|---:|---:|---:|
| downslope | 7 | 510 | 0 | 0.000000 |
| flat | 1 | 481 | 481 | 1.000000 |
| flat | 7 | 1 | 0 | 0.000000 |
| goal_flat | 7 | 491 | 0 | 0.000000 |
| rough | 5 | 1 | 0 | 0.000000 |
| rough | 7 | 427 | 0 | 0.000000 |
| unknown | 7 | 8 | 0 | 0.000000 |
| upslope | 1 | 1 | 1 | 1.000000 |
| upslope | 5 | 428 | 428 | 1.000000 |

## Numeric summary: all vs accepted

| column | all mean | all min | all max | accepted mean | accepted min | accepted max |
|---|---:|---:|---:|---:|---:|---:|
| emp_vx | 0.169758 | 0.025000 | 0.210000 | 0.210000 | 0.210000 | 0.210000 |
| proj_vx | 0.139301 | 0.025000 | 0.220000 | 0.211346 | 0.201600 | 0.220000 |
| out_vx | 0.131193 | 0.025000 | 0.220000 | 0.211346 | 0.201600 | 0.220000 |
| delta_vx | -0.030440 | -0.177500 | 0.010000 | 0.001346 | -0.008400 | 0.010000 |
| emp_yaw_rate | 0.001685 | -0.002468 | 0.006493 | 0.000978 | -0.001856 | 0.003322 |
| proj_yaw_rate | 0.001303 | -0.001856 | 0.004545 | 0.000927 | -0.001856 | 0.003156 |
| out_yaw_rate | 0.001103 | -0.001856 | 0.004842 | 0.000927 | -0.001856 | 0.003156 |
| delta_yaw_rate | -0.000383 | -0.001978 | 0.002468 | -0.000051 | -0.000204 | 0.000128 |
| emp_body_h | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| proj_body_h | 0.315105 | 0.312000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| out_body_h | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| delta_body_h | -0.004895 | -0.008000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| emp_clearance | 0.046819 | 0.045000 | 0.055000 | 0.045000 | 0.045000 | 0.045000 |
| proj_clearance | 0.051829 | 0.042000 | 0.063000 | 0.045292 | 0.042000 | 0.049000 |
| out_clearance | 0.046932 | 0.042000 | 0.055000 | 0.045292 | 0.042000 | 0.049000 |
| delta_clearance | 0.005010 | -0.003000 | 0.018000 | 0.000292 | -0.003000 | 0.004000 |
| emp_age_s | 0.154617 | 0.001636 | 8.306598 | 0.006207 | 0.001636 | 0.007188 |
| proj_age_s | 0.056176 | 0.053086 | 0.057472 | 0.056221 | 0.053522 | 0.057472 |
| theta_age_s | 21.405320 | 0.086341 | 100.089323 | 0.089535 | 0.086341 | 0.093744 |

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
