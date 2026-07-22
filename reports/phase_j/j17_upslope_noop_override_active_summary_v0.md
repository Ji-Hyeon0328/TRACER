# TRACER Phase-J17 Upslope No-Op Override Active Summary v0

J17 explicitly overrides upslope context to action 5 no-op before J4 projection.

## Purpose

J16 showed severe regression with the original upslope action 5.  
J17 tests whether upslope active routing is stable when action 5 is made near no-op.

## Gate-level result

- J7 log: `logs/phase_j/j7_for_j17_upslope_noop_override_active_20260722_015310/guarded_meta_action_ref_gate_j7_v0.csv`
- J7 report: `reports/phase_j/j7_for_j17_upslope_noop_override_active_20260722_015310_check_v0.md`
- J17 override log: `logs/phase_j/j17_upslope_noop_override_active_20260722_015251/upslope_noop_override_j17_v0.csv`
- D5 log dir: `/home/kraken/Tracer/TRACER/logs/phase_d5_shadow_20260722_015322_trial_1`

| context | accepted / rows | accept rate |
|---|---:|---:|
| upslope | 431 / 431 | 1.000 |
| flat | 0 / 443 | 0.000 |
| rough | 0 / 433 | 0.000 |
| downslope | 0 / 518 | 0.000 |
| goal_flat | 0 / 700 | 0.000 |
| unknown | 0 / 7 | 0.000 |

## Upslope override behavior

| context | input action | output action | override | rows |
|---|---:|---:|---:|---:|
| upslope | 5 | 5 | 1 | 431 |

## Accepted upslope command

| metric | empirical | projected/output | delta |
|---|---:|---:|---:|
| vx | 0.210 | 0.210023 | +0.000023 |
| body_h | 0.320 | 0.320 | 0.000000 |
| clearance | 0.045 | 0.044993 | -0.000007 |

Output safety violations were all zero.

## Mission-level result

| metric | value |
|---|---:|
| final_x | 8.116892 |
| final_y | 0.398084 |
| max_x | 8.141550 |
| max_abs_y | 0.443260 |
| mean_abs_y | 0.215593 |
| goal_reached_x8 | True |

## Interpretation

J17 improved substantially compared with J16, but it did not recover to the J13/J15B baseline-quality trajectory.

Evidence so far:

- J15B flat no-op active route was stable.
- J16 original upslope action 5 regressed severely.
- J17 upslope no-op improved over J16 but still showed elevated lateral drift.

Therefore, the original upslope action 5 value is problematic, and upslope active routing or segment-transition timing may also need further inspection.

## Decision

Do not promote upslope active routing yet.  
Next step should be a conservative combined no-op probe or an immediate baseline repeat to check whether the J17 drift is active-route-specific or run-variance-related.
