# TRACER Phase-J7 Guarded Ref Gate Log Check v0

- csv: `logs/phase_j/j7_for_j10_downslope_shadow_20260721_221843/guarded_meta_action_ref_gate_j7_v0.csv`
- rows: `2524`
- accepted rows: `1071`
- accept rate: `0.424326`
- first_t: `1784686797.426966`
- last_t: `1784687049.726912`

## Context counts

| context | rows |
|---|---:|
| downslope | 531 |
| flat | 506 |
| goal_flat | 677 |
| rough | 384 |
| unknown | 1 |
| upslope | 425 |

## Decision reasons

| reason | rows |
|---|---:|
| accepted_guarded | 1071 |
| delta_vx_too_large | 747 |
| hold_or_goal_protected | 443 |
| stale_empirical_ref | 260 |
| context_not_allowed_for_first_active | 2 |
| missing_projected_ref | 1 |

## Output source

| source | rows |
|---|---:|
| empirical | 1192 |
| projected | 1071 |
| failsafe_hold | 261 |

## Acceptance by context

| context | rows | accepted | accept rate |
|---|---:|---:|---:|
| downslope | 531 | 141 | 0.265537 |
| flat | 506 | 505 | 0.998024 |
| goal_flat | 677 | 0 | 0.000000 |
| rough | 384 | 0 | 0.000000 |
| unknown | 1 | 0 | 0.000000 |
| upslope | 425 | 425 | 1.000000 |

## Acceptance by context/action

| context | action_id | rows | accepted | accept rate |
|---|---:|---:|---:|---:|
| downslope | 6 | 142 | 141 | 0.992958 |
| downslope | 7 | 363 | 0 | 0.000000 |
| downslope | 8 | 26 | 0 | 0.000000 |
| flat | 1 | 506 | 505 | 0.998024 |
| goal_flat | 8 | 677 | 0 | 0.000000 |
| rough | 5 | 1 | 0 | 0.000000 |
| rough | 7 | 383 | 0 | 0.000000 |
| unknown | 7 | 1 | 0 | 0.000000 |
| upslope | 1 | 1 | 1 | 1.000000 |
| upslope | 5 | 424 | 424 | 1.000000 |

## Numeric summary: all vs accepted

| column | all mean | all min | all max | accepted mean | accepted min | accepted max |
|---|---:|---:|---:|---:|---:|---:|
| emp_vx | 0.158884 | 0.025000 | 0.210000 | 0.209013 | 0.202500 | 0.210000 |
| proj_vx | 0.134311 | 0.025000 | 0.220000 | 0.207230 | 0.178200 | 0.220000 |
| out_vx | 0.158054 | 0.025000 | 0.220000 | 0.207230 | 0.178200 | 0.220000 |
| delta_vx | -0.024552 | -0.177500 | 0.010000 | -0.001783 | -0.024300 | 0.010000 |
| emp_yaw_rate | 0.007502 | -0.002074 | 0.022123 | 0.005234 | -0.002074 | 0.014434 |
| proj_yaw_rate | 0.005735 | -0.002074 | 0.015486 | 0.004913 | -0.002074 | 0.013697 |
| out_yaw_rate | 0.007366 | -0.002074 | 0.022123 | 0.004913 | -0.002074 | 0.013697 |
| delta_yaw_rate | -0.001770 | -0.006676 | 0.000173 | -0.000321 | -0.001145 | 0.000173 |
| emp_body_h | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| proj_body_h | 0.317406 | 0.312000 | 0.320000 | 0.319473 | 0.316000 | 0.320000 |
| out_body_h | 0.319777 | 0.316000 | 0.320000 | 0.319473 | 0.316000 | 0.320000 |
| delta_body_h | -0.002594 | -0.008000 | 0.000000 | -0.000527 | -0.004000 | 0.000000 |
| emp_clearance | 0.046521 | 0.045000 | 0.055000 | 0.045000 | 0.045000 | 0.045000 |
| proj_clearance | 0.049187 | 0.042000 | 0.063000 | 0.045686 | 0.042000 | 0.049000 |
| out_clearance | 0.046813 | 0.042000 | 0.055000 | 0.045686 | 0.042000 | 0.049000 |
| delta_clearance | 0.002665 | -0.006000 | 0.018000 | 0.000686 | -0.003000 | 0.004000 |
| emp_age_s | 1.452316 | 0.031717 | 26.734031 | 0.034949 | 0.032593 | 0.040541 |
| proj_age_s | 0.081812 | 0.075947 | 0.381247 | 0.081598 | 0.075947 | 0.086874 |
| theta_age_s | 0.079452 | 0.072364 | 0.877668 | 0.078140 | 0.072364 | 0.083928 |

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
