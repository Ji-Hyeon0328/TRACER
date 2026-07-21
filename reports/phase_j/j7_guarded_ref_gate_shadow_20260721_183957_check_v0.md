# TRACER Phase-J7 Guarded Ref Gate Log Check v0

- csv: `logs/phase_j/j7_guarded_ref_gate_shadow_20260721_183957/guarded_meta_action_ref_gate_j7_v0.csv`
- rows: `2635`
- accepted rows: `885`
- accept rate: `0.335863`
- first_t: `1784673666.844902`
- last_t: `1784673930.245033`

## Context counts

| context | rows |
|---|---:|
| downslope | 503 |
| flat | 469 |
| goal_flat | 799 |
| rough | 444 |
| unknown | 3 |
| upslope | 417 |

## Decision reasons

| reason | rows |
|---|---:|
| delta_vx_too_large | 924 |
| accepted_guarded | 885 |
| stale_empirical_ref | 462 |
| hold_or_goal_protected | 362 |
| context_not_allowed_for_first_active | 1 |
| goal_x_protected | 1 |

## Output source

| source | rows |
|---|---:|
| empirical | 1750 |
| projected | 885 |

## Acceptance by context

| context | rows | accepted | accept rate |
|---|---:|---:|---:|
| downslope | 503 | 0 | 0.000000 |
| flat | 469 | 468 | 0.997868 |
| goal_flat | 799 | 0 | 0.000000 |
| rough | 444 | 0 | 0.000000 |
| unknown | 3 | 0 | 0.000000 |
| upslope | 417 | 417 | 1.000000 |

## Acceptance by context/action

| context | action_id | rows | accepted | accept rate |
|---|---:|---:|---:|---:|
| downslope | 6 | 478 | 0 | 0.000000 |
| downslope | 8 | 25 | 0 | 0.000000 |
| flat | 1 | 468 | 468 | 1.000000 |
| flat | 7 | 1 | 0 | 0.000000 |
| goal_flat | 8 | 799 | 0 | 0.000000 |
| rough | 5 | 1 | 0 | 0.000000 |
| rough | 7 | 443 | 0 | 0.000000 |
| unknown | 7 | 3 | 0 | 0.000000 |
| upslope | 1 | 1 | 1 | 1.000000 |
| upslope | 5 | 416 | 416 | 1.000000 |

## Numeric summary: all vs accepted

| column | all mean | all min | all max | accepted mean | accepted min | accepted max |
|---|---:|---:|---:|---:|---:|---:|
| emp_vx | 0.152507 | 0.025000 | 0.210000 | 0.210000 | 0.210000 | 0.210000 |
| proj_vx | 0.130091 | 0.025000 | 0.220000 | 0.211351 | 0.201600 | 0.220000 |
| out_vx | 0.152960 | 0.025000 | 0.220000 | 0.211351 | 0.201600 | 0.220000 |
| delta_vx | -0.022416 | -0.177500 | 0.010000 | 0.001351 | -0.008400 | 0.010000 |
| emp_yaw_rate | -0.001120 | -0.004890 | 0.003990 | -0.002910 | -0.004890 | 0.000003 |
| proj_yaw_rate | -0.001043 | -0.004645 | 0.002793 | -0.002820 | -0.004645 | 0.000003 |
| out_yaw_rate | -0.001090 | -0.004697 | 0.003990 | -0.002820 | -0.004645 | 0.000003 |
| delta_yaw_rate | 0.000078 | -0.001197 | 0.004697 | 0.000090 | 0.000000 | 0.000244 |
| emp_body_h | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| proj_body_h | 0.317554 | 0.312000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| out_body_h | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| delta_body_h | -0.002446 | -0.008000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| emp_clearance | 0.046681 | 0.045000 | 0.055000 | 0.045000 | 0.045000 | 0.045000 |
| proj_clearance | 0.048863 | 0.042000 | 0.063000 | 0.045290 | 0.042000 | 0.049000 |
| out_clearance | 0.046779 | 0.042000 | 0.055000 | 0.045290 | 0.042000 | 0.049000 |
| delta_clearance | 0.002182 | -0.003000 | 0.008000 | 0.000290 | -0.003000 | 0.004000 |
| emp_age_s | 4.301982 | 0.077316 | 47.183587 | 0.083621 | 0.077316 | 0.086208 |
| proj_age_s | 0.024969 | 0.020648 | 0.524624 | 0.024523 | 0.021273 | 0.027430 |
| theta_age_s | 0.079703 | 0.049602 | 3.753191 | 0.053190 | 0.050213 | 0.055841 |

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
