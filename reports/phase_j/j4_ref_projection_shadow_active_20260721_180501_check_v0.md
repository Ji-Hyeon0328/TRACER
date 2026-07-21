# TRACER Phase-J4 Meta-Action Ref Projection Shadow Log Check v0

- csv: `logs/phase_j/j4_ref_projection_shadow_active_20260721_180501/meta_action_ref_projection_shadow_j4_v0.csv`
- rows: `4832`
- first_t: `1784671570.797999`
- last_t: `1784672053.898491`

## Context counts

| context | rows |
|---|---:|
| downslope | 493 |
| flat | 482 |
| goal_flat | 2983 |
| rough | 431 |
| unknown | 7 |
| upslope | 436 |

## Action id distribution

| action_id | rows |
|---:|---:|
| 8 | 3004 |
| 1 | 483 |
| 6 | 472 |
| 7 | 438 |
| 5 | 435 |

## Action id by context

| context | action_id | rows |
|---|---:|---:|
| downslope | 6 | 472 |
| downslope | 8 | 21 |
| flat | 1 | 482 |
| goal_flat | 8 | 2983 |
| rough | 7 | 431 |
| unknown | 7 | 7 |
| upslope | 5 | 435 |
| upslope | 1 | 1 |

## Numeric summary

| column | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| emp_vx | 4832 | 0.094984 | 0.088488 | 0.025000 | 0.210000 |
| proj_vx | 4832 | 0.083194 | 0.077536 | 0.025000 | 0.220000 |
| delta_vx | 4832 | -0.011790 | 0.027329 | -0.177500 | 0.010000 |
| emp_yaw_rate | 4832 | 0.001037 | 0.001822 | -0.001360 | 0.005758 |
| proj_yaw_rate | 4832 | 0.000816 | 0.001445 | -0.001360 | 0.004483 |
| delta_yaw_rate | 4832 | -0.000221 | 0.000534 | -0.003891 | 0.000000 |
| emp_body_h | 4832 | 0.320000 | 0.000000 | 0.320000 | 0.320000 |
| proj_body_h | 4832 | 0.318689 | 0.002757 | 0.312000 | 0.320000 |
| delta_body_h | 4832 | -0.001311 | 0.002757 | -0.008000 | 0.000000 |
| emp_clearance | 4832 | 0.045892 | 0.002850 | 0.045000 | 0.055000 |
| proj_clearance | 4832 | 0.047068 | 0.005350 | 0.042000 | 0.063000 |
| delta_clearance | 4832 | 0.001176 | 0.002885 | -0.003000 | 0.008000 |
| proj_enable | 4832 | 1.000000 | 0.000000 | 1.000000 | 1.000000 |
| hold_guard | 4832 | 0.621689 | 0.484966 | 0.000000 | 1.000000 |

## Safety projection violations

| check | count |
|---|---:|
| vx_low | 0 |
| vx_high | 0 |
| yaw_low | 0 |
| yaw_high | 0 |
| body_h_low | 0 |
| body_h_high | 0 |
| clearance_low | 0 |
| clearance_high | 0 |

## Quick interpretation

- J4 is shadow-only and should not modify `/tracer/mpc_reference`.
- Expected: projected refs remain inside safety bounds.
- Expected: flat action increases vx slightly, rough/downslope reduce vx or add stability/clearance, goal_flat uses hold guard.
