# TRACER Phase-J10C Conservative Downslope Override Shadow Summary v0

J10C validates a context-action override that forces downslope meta-action theta to conservative action 6 before J4 projection.

## Result

- J7 log: `logs/phase_j/j7_for_j10c_override_shadow_20260721_233818/guarded_meta_action_ref_gate_j7_v0.csv`
- J7 report: `reports/phase_j/j7_for_j10c_override_shadow_20260721_233818_check_v0.md`
- J10C log: `logs/phase_j/j10c_context_action_override_shadow_20260721_233802/context_action_override_j10c_v0.csv`

## Acceptance by context

| context | accepted / rows | accept rate |
|---|---:|---:|
| downslope | 507 / 532 | 0.953 |
| flat | 474 / 476 | 0.996 |
| upslope | 427 / 427 | 1.000 |
| rough | 0 / 434 | 0.000 |
| goal_flat | 0 / 4262 | 0.000 |
| unknown | 0 / 10 | 0.000 |

## Downslope override behavior

| context | input action | output action | override | rows |
|---|---:|---:|---:|---:|
| downslope | 7 | 6 | 1 | 284 |
| downslope | 6 | 6 | 1 | 224 |
| downslope | 8 | 6 | 1 | 23 |

## Accepted command range

| metric | accepted min | accepted mean | accepted max |
|---|---:|---:|---:|
| out_vx | 0.1782 | 0.1994 | 0.2200 |
| delta_vx | -0.0243 | -0.0079 | 0.0100 |
| out_body_h | 0.3160 | 0.3186 | 0.3200 |
| out_clearance | 0.0420 | 0.0466 | 0.0490 |

## Safe interpretation

J10C passed shadow validation. The next active candidate is J11:

- active: flat, upslope, conservative downslope
- protected: rough, unknown, goal_flat
- safety violations: zero in shadow validation
