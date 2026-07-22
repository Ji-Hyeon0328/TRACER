# TRACER Phase-J11 Conservative Downslope Active Audit Summary v0

J11 tested guarded active control with:

- active projected references in flat
- active projected references in upslope
- active conservative downslope references through J10C override
- protected rough/unknown/goal_flat

## Gate-level result

- J7 log: `logs/phase_j/j7_for_j11_active_20260721_235902/guarded_meta_action_ref_gate_j7_v0.csv`
- J7 report: `reports/phase_j/j7_for_j11_active_20260721_235902_check_v0.md`
- D4/D5 manifest: `reports/phase_d4_context_meta_repeat_20260722_000137_manifest.tsv`
- D5 log dir: `/home/kraken/Tracer/TRACER/logs/phase_d5_shadow_20260722_000137_trial_1`

| context | accepted / rows | accept rate |
|---|---:|---:|
| downslope | 545 / 571 | 0.954 |
| flat | 433 / 434 | 0.998 |
| upslope | 444 / 444 | 1.000 |
| rough | 0 / 442 | 0.000 |
| goal_flat | 0 / 5156 | 0.000 |
| unknown | 0 / 8 | 0.000 |

Output safety violations were all zero.

## Mission-level result

| metric | value |
|---|---:|
| final_x | 8.141307 |
| final_y | -0.298542 |
| max_x | 8.145411 |
| max_abs_y | 0.447223 |
| mean_abs_y | 0.265798 |
| goal_reached_x8 | True |

## Interpretation

J11 passed the gate-level safety checks and reached the goal, but the mission-level trajectory quality regressed.

Compared with the previous J9 comparison:

- J8 guarded active max_abs_y was about 0.228
- J9 baseline max_abs_y was about 0.435
- J11 max_abs_y was about 0.447

Therefore J11 should not be treated as an improved active controller. It is an audit result showing that conservative downslope opening is not sufficient by itself and may interact poorly with earlier active segments or run-to-run drift.

## Decision

- Keep J8 as the current best active configuration.
- Do not promote J11 to the default active stack.
- Next step should be ablation:
  - reproduce J8 active once more,
  - test flat-only / upslope-only / downslope-only active if needed,
  - inspect segment-wise drift before enabling downslope active again.
