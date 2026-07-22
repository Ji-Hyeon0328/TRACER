# TRACER Phase-J8 Guarded Active Mission Outcome v0

This report summarizes the mission outcome for the first guarded active meta-action gate smoke test.

- manifest: `reports/phase_d4_context_meta_repeat_20260721_185056_manifest.tsv`
- policy input csv: `/home/kraken/Tracer/TRACER/logs/phase_d5_shadow_20260721_185056_trial_1/policy_inputs_v0.csv`
- J7 gate report: `reports/phase_j/j7_guarded_ref_gate_active_20260721_185021_check_v0.md`
- rows: `2128`
- valid xy rows: `2128`

## Mission outcome

| metric | value |
|---|---:|
| final_x | 8.171026635605655 |
| final_y | -0.10334065536997024 |
| max_x | 8.171026635605655 |
| min_x | 0.10687141235105203 |
| max_abs_y | 0.22821334331921853 |
| goal_reached_x8 | True |

## Context counts

| context | rows |
|---|---:|
| downslope | 485 |
| upslope | 455 |
| rough | 452 |
| flat | 392 |
| goal_flat | 344 |

## Safe interpretation

- J8 used the guarded active path, routing J7 output to `/tracer/mpc_reference`.
- J7 accepted projected references only in flat/upslope and rejected rough/downslope/unknown/goal rows.
- This report checks whether the robot still completed the mission-level goal under that guarded active routing.
- If `goal_reached_x8` is true and lateral drift is acceptable, this is a valid first guarded active smoke result.
