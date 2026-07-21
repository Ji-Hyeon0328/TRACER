# TRACER Phase-J6 Guarded Active Dry-Run v0

This evaluates whether J4 projected references are safe candidates for a future guarded active test.

- input csv: `logs/phase_j/j4_ref_projection_shadow_active_20260721_180501/meta_action_ref_projection_shadow_j4_v0.csv`
- output csv: `datasets/phase_j/j6_guarded_active_dryrun_20260721_180501_v0.csv`
- rows: `10573`
- accepted rows: `918`
- accept rate: `0.086825`
- allowed contexts: `flat, upslope, downslope`

## Guard thresholds

- max_abs_delta_vx: `0.03`
- max_abs_delta_yaw: `0.02`
- max_abs_delta_body_h: `0.006`
- max_abs_delta_clearance: `0.006`
- min_active_body_h: `0.314`
- min_rough_active_vx: `0.16`
- min_downslope_active_vx: `0.15`

## Decision reasons

| reason | rows |
|---|---:|
| hold_or_goal_protected | 8745 |
| accepted_dryrun | 918 |
| delta_vx_too_large | 910 |

## Acceptance by context

| context | rows | accepted | accept rate |
|---|---:|---:|---:|
| downslope | 493 | 0 | 0.000000 |
| flat | 482 | 482 | 1.000000 |
| goal_flat | 8724 | 0 | 0.000000 |
| rough | 431 | 0 | 0.000000 |
| unknown | 7 | 0 | 0.000000 |
| upslope | 436 | 436 | 1.000000 |

## Acceptance by context/action

| context | action_id | rows | accepted | accept rate |
|---|---:|---:|---:|---:|
| downslope | 6 | 472 | 0 | 0.000000 |
| downslope | 8 | 21 | 0 | 0.000000 |
| flat | 1 | 482 | 482 | 1.000000 |
| goal_flat | 8 | 8724 | 0 | 0.000000 |
| rough | 7 | 431 | 0 | 0.000000 |
| unknown | 7 | 7 | 0 | 0.000000 |
| upslope | 1 | 1 | 1 | 1.000000 |
| upslope | 5 | 435 | 435 | 1.000000 |

## Numeric summary: all vs accepted

| column | all mean | all min | all max | accepted mean | accepted min | accepted max |
|---|---:|---:|---:|---:|---:|---:|
| delta_vx | -0.005388 | -0.177500 | 0.010000 | 0.001281 | -0.008400 | 0.010000 |
| delta_yaw_rate | -0.000101 | -0.003891 | 0.000000 | -0.000084 | -0.000236 | 0.000000 |
| delta_body_h | -0.000599 | -0.008000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| delta_clearance | 0.000538 | -0.003000 | 0.008000 | 0.000317 | -0.003000 | 0.004000 |
| proj_vx | 0.051596 | 0.025000 | 0.220000 | 0.211281 | 0.201600 | 0.220000 |
| proj_body_h | 0.319401 | 0.312000 | 0.320000 | 0.320000 | 0.320000 | 0.320000 |
| proj_clearance | 0.045945 | 0.042000 | 0.063000 | 0.045317 | 0.042000 | 0.049000 |

## Safe interpretation

- J6 is a dry-run gate only and does not modify active control.
- First active candidate should only use rows/actions that pass this dry-run gate.
- Goal/hold rows are protected and rejected by design.
- Rough/unknown aggressive recovery projections are rejected for the first active test unless a later explicit experiment enables them.
