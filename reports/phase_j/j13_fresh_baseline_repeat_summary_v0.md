# TRACER Phase-J13 Fresh Baseline Repeat Summary v0

J13 repeats the baseline D7/D6 gated controller with no J3/J4/J7 active meta-action routing.

- manifest: `reports/phase_d4_context_meta_repeat_20260722_003612_manifest.tsv`
- policy input csv: `/home/kraken/Tracer/TRACER/logs/phase_d5_shadow_20260722_003612_trial_1/policy_inputs_v0.csv`
- D5 gate csv: `/home/kraken/Tracer/TRACER/logs/phase_d5_shadow_20260722_003612_trial_1/gated_selector_dryrun_v0.csv`

## Mission-level result

| metric | value |
|---|---:|
| final_x | 8.030218 |
| final_y | 0.134797 |
| max_x | 8.073839 |
| max_abs_y | 0.233972 |
| mean_abs_y | 0.111921 |
| goal_reached_x8 | True |

## Context counts

| context | rows |
|---|---:|
| downslope | 520 |
| rough | 440 |
| flat | 439 |
| upslope | 438 |
| goal_flat | 390 |

## D5 gate summary

| metric | value |
|---|---:|
| gate rows | 2228 |
| gate accepted | 1821 |
| gate accept rate | 0.817325 |

| selected_source | rows |
|---|---:|
| learned_selected_dims | 1821 |
| empirical | 407 |

## Interpretation guide

- If J13 baseline max_abs_y is also high, the recent regressions may be dominated by reset/controller/run-to-run variance.
- If J13 baseline max_abs_y is moderate while J12 was high, the active flat/upslope projection is likely contributing to lateral drift.
- Do not promote J8/J12 active routing until reproducibility is established.
