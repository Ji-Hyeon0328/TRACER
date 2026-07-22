# TRACER Phase-J12 J8-Repeat Active Reproducibility Audit Summary v0

J12 repeated the J8-style guarded active configuration:

- active projected references in flat
- active projected references in upslope
- protected downslope, rough, unknown, and goal_flat

## Gate-level result

- J7 log: `logs/phase_j/j7_for_j12_j8_repeat_active_20260722_002249/guarded_meta_action_ref_gate_j7_v0.csv`
- J7 report: `reports/phase_j/j7_for_j12_j8_repeat_active_20260722_002249_check_v0.md`
- D4/D5 manifest: `reports/phase_d4_context_meta_repeat_20260722_002308_manifest.tsv`
- D5 log dir: `/home/kraken/Tracer/TRACER/logs/phase_d5_shadow_20260722_002308_trial_1`

| context | accepted / rows | accept rate |
|---|---:|---:|
| flat | 425 / 426 | 0.998 |
| upslope | 450 / 450 | 1.000 |
| downslope | 0 / 504 | 0.000 |
| rough | 0 / 439 | 0.000 |
| goal_flat | 0 / 1736 | 0.000 |
| unknown | 0 / 1 | 0.000 |

Output safety violations were all zero.

## Mission-level result

| metric | value |
|---|---:|
| final_x | 8.174878 |
| final_y | -0.615806 |
| max_x | 8.174878 |
| max_abs_y | 0.842522 |
| mean_abs_y | 0.454851 |
| goal_reached_x8 | True |

## Interpretation

J12 reached the goal and passed gate-level safety checks, but the trajectory quality regressed severely.

Compared with earlier runs:

- J8 guarded active max_abs_y was about 0.228
- J9 baseline max_abs_y was about 0.435
- J11 conservative-downslope active max_abs_y was about 0.447
- J12 J8-repeat active max_abs_y was about 0.843

Therefore, the J8 active result is not yet reproducible. The active flat/upslope projection should remain experimental and should not be promoted to a default controller.

## Decision

- Keep J8 as a promising single-run result only.
- Keep J11 and J12 as audit/regression evidence.
- Do not open rough or downslope active further.
- Next step: run a fresh baseline repeat to separate controller/environment variance from active projection effects.
