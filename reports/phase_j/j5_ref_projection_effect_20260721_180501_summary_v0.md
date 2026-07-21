# TRACER Phase-J5 Ref Projection Effect Summary v0

This summarizes how the J4 shadow meta-action projection changes empirical references by context and action.

- input csv: `logs/phase_j/j4_ref_projection_shadow_active_20260721_180501/meta_action_ref_projection_shadow_j4_v0.csv`
- output csv: `datasets/phase_j/j5_ref_projection_effect_20260721_180501_v0.csv`
- rows: `6368`

## Overall

| rows | emp vx | proj vx | delta vx | emp clearance | proj clearance | delta clearance | hold rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6368 | 0.078104 | 0.069157 | -0.008946 | 0.045677 | 0.046569 | 0.000892 | 0.712940 |

## By context

| context | rows | emp vx | proj vx | delta vx | proj body_h | delta body_h | proj clearance | delta clearance | hold rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| downslope | 493 | 0.202500 | 0.152287 | -0.050213 | 0.314256 | -0.005744 | 0.048830 | 0.003830 | 0.042596 |
| flat | 482 | 0.210000 | 0.220000 | 0.010000 | 0.320000 | 0.000000 | 0.042000 | -0.003000 | 0.000000 |
| goal_flat | 4519 | 0.025432 | 0.025000 | -0.000432 | 0.320000 | 0.000000 | 0.045000 | 0.000000 | 1.000000 |
| rough | 431 | 0.205000 | 0.133250 | -0.071750 | 0.312000 | -0.008000 | 0.063000 | 0.008000 | 0.000000 |
| unknown | 7 | 0.210000 | 0.136500 | -0.073500 | 0.312000 | -0.008000 | 0.053000 | 0.008000 | 0.000000 |
| upslope | 436 | 0.210000 | 0.201642 | -0.008358 | 0.320000 | 0.000000 | 0.048984 | 0.003984 | 0.000000 |

## By context/action

| context | action_id | rows | emp vx | proj vx | delta vx | proj body_h | proj clearance | delta clearance | hold rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| downslope | 6 | 472 | 0.202500 | 0.157950 | -0.044550 | 0.314000 | 0.049000 | 0.004000 | 0.000000 |
| downslope | 8 | 21 | 0.202500 | 0.025000 | -0.177500 | 0.320000 | 0.045000 | 0.000000 | 1.000000 |
| flat | 1 | 482 | 0.210000 | 0.220000 | 0.010000 | 0.320000 | 0.042000 | -0.003000 | 0.000000 |
| goal_flat | 8 | 4519 | 0.025432 | 0.025000 | -0.000432 | 0.320000 | 0.045000 | 0.000000 | 1.000000 |
| rough | 7 | 431 | 0.205000 | 0.133250 | -0.071750 | 0.312000 | 0.063000 | 0.008000 | 0.000000 |
| unknown | 7 | 7 | 0.210000 | 0.136500 | -0.073500 | 0.312000 | 0.053000 | 0.008000 | 0.000000 |
| upslope | 1 | 1 | 0.210000 | 0.220000 | 0.010000 | 0.320000 | 0.042000 | -0.003000 | 0.000000 |
| upslope | 5 | 435 | 0.210000 | 0.201600 | -0.008400 | 0.320000 | 0.049000 | 0.004000 | 0.000000 |

## Safe interpretation

- J5 is an offline summary of J4 shadow projection only.
- It does not modify active control.
- Good signs: flat slightly increases vx, rough/downslope reduce vx or increase stability margin, goal_flat holds, and all projected refs remain within safety bounds.
- If these projected references look reasonable, the next step is not full active control yet, but a guarded active candidate design.
