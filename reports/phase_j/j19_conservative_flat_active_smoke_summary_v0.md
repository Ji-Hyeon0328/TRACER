# TRACER Phase-J19 Conservative Flat Active Smoke Summary v0

J19 tested a deployable conservative flat-only active profile:

- J3 meta-action shadow
- J15B flat no-op override
- J4 projected reference
- J7 guarded active output to `/tracer/mpc_reference`
- D5 gated output diverted to shadow topic

## Gate-level result

- J19 log root: `logs/phase_j/j19_conservative_flat_active_20260722_114201`
- J7 report: `reports/phase_j/j19_conservative_flat_active_smoke_j7_check_v0.md`
- D4/D5 manifest: `reports/phase_d4_context_meta_repeat_20260722_114223_manifest.tsv`
- D5 log dir: `/home/kraken/Tracer/TRACER/logs/phase_d5_shadow_20260722_114223_trial_1`

| context | accepted / rows | accept rate |
|---|---:|---:|
| flat | 444 / 445 | 0.998 |
| upslope | 0 / 424 | 0.000 |
| rough | 0 / 446 | 0.000 |
| downslope | 0 / 505 | 0.000 |
| goal_flat | 0 / 1425 | 0.000 |
| unknown | 0 / 7 | 0.000 |

## Accepted flat command

| metric | empirical | projected/output | delta |
|---|---:|---:|---:|
| vx | 0.210 | 0.210 | 0.000 |
| body_h | 0.320 | 0.320 | 0.000 |
| clearance | 0.045 | 0.045 | 0.000 |

Output safety violations were all zero.

## Mission-level result

| metric | value |
|---|---:|
| final_x | 7.980386 |
| final_y | -0.627671 |
| max_x | 8.078257 |
| max_abs_y | 0.669793 |
| mean_abs_y | 0.214920 |
| goal_reached_x8 | True |

## Interpretation

J19 passed the gate-level checks, but mission-level trajectory quality regressed.

This means the conservative flat-only deployable profile should not be promoted yet.  
The result is inconsistent with the stronger J18 flat_noop repeat trend, so this should be treated as a bad smoke run or deployment-profile variance case.

## Decision

- Do not tag J19 as safe active candidate.
- Keep J18 flat_noop as promising evidence.
- Next step: run an exact J19-profile N-repeat, or compare J19 profile against baseline in an interleaved repeat.
