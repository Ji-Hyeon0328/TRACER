# TRACER Phase-J9 Baseline vs J8 Guarded Active Comparison v0

This compares the standard D7/D6 gated baseline against the first J8 guarded-active meta-action smoke test.

- baseline manifest: `reports/phase_d4_context_meta_repeat_20260721_220205_manifest.tsv`
- active manifest: `reports/phase_d4_context_meta_repeat_20260721_185056_manifest.tsv`
- active J7 csv: `logs/phase_j/j7_guarded_ref_gate_active_20260721_185021/guarded_meta_action_ref_gate_j7_v0.csv`
- output csv: `datasets/phase_j/j9_baseline_vs_j8_active_comparison_v0.csv`

## Mission-level outcome

| metric | baseline | J8 guarded active | delta active-baseline |
|---|---:|---:|---:|
| final_x | 8.196250 | 8.171027 | -0.025224 |
| final_y | -0.342269 | -0.103341 | 0.238928 |
| max_x | 8.197320 | 8.171027 | -0.026294 |
| mean_abs_y | 0.171348 | 0.096507 | -0.074841 |
| max_abs_y | 0.435303 | 0.228213 | -0.207090 |
| duration_s | 212.400254 | 212.700089 | 0.299836 |
| time_to_goal_s | 179.500151 | 178.399896 | -1.100255 |
| goal_reached | True | True | NA |

## Context counts

| context | baseline rows | active rows |
|---|---:|---:|
| downslope | 484 | 485 |
| flat | 437 | 392 |
| goal_flat | 330 | 344 |
| rough | 423 | 452 |
| unknown | 0 | 0 |
| upslope | 451 | 455 |

## Segment-level lateral metrics

| context | baseline mean | active mean | delta mean | baseline max | active max | delta max |
|---|---:|---:|---:|---:|---:|---:|
| downslope | 0.234562 | 0.132744 | -0.101818 | 0.377955 | 0.213965 | -0.163990 |
| flat | 0.023857 | 0.047115 | 0.023258 | 0.079451 | 0.081144 | 0.001693 |
| goal_flat | 0.407522 | 0.135849 | -0.271673 | 0.435303 | 0.165087 | -0.270216 |
| rough | 0.114268 | 0.100688 | -0.013580 | 0.171066 | 0.228213 | 0.057148 |
| unknown | NA | NA | NA | NA | NA | NA |
| upslope | 0.127149 | 0.066537 | -0.060611 | 0.168919 | 0.089543 | -0.079376 |

## Command-level comparison by context

| context | baseline cmd vx | active out vx | baseline clr | active out clr | base gate accept | J7 accept | projected rate | failsafe rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| downslope | 0.204852 | 0.202500 | 0.045016 | 0.045000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| flat | 0.211340 | 0.219976 | 0.045007 | 0.042007 | 0.977064 | 0.997619 | 0.997619 | 0.000000 |
| goal_flat | 0.031435 | 0.028008 | 0.045000 | 0.045000 | 0.000000 | 0.000000 | 0.000000 | 0.405085 |
| rough | 0.199284 | 0.205011 | 0.054509 | 0.054978 | 0.997642 | 0.000000 | 0.000000 | 0.000000 |
| unknown | NA | 0.193182 | NA | 0.045000 | NA | 0.000000 | 0.000000 | 0.090909 |
| upslope | 0.202224 | 0.201641 | 0.044728 | 0.048985 | 0.986696 | 1.000000 | 1.000000 | 0.000000 |

## Source / reason counts

### Baseline D5 selected source

| source | rows |
|---|---:|
| learned_selected_dims | 1777 |
| empirical | 348 |

### J8 J7 output source

| source | rows |
|---|---:|
| empirical | 1299 |
| projected | 873 |
| failsafe_hold | 240 |

### J8 J7 decision reason

| reason | rows |
|---|---:|
| delta_vx_too_large | 927 |
| accepted_guarded | 873 |
| hold_or_goal_protected | 369 |
| stale_empirical_ref | 239 |
| context_not_allowed_for_first_active | 2 |
| missing_projected_ref | 1 |
| goal_x_protected | 1 |

## Safe interpretation

- J9 is not claiming global improvement yet; it checks whether J8 active intervention improves or preserves mission behavior.
- Flat/upslope are intervention segments where projected references were allowed.
- Rough/downslope/goal are protected segments where the desired result is non-degradation.
- If J8 reaches the goal with comparable or lower lateral drift, the next step is conservative rough/downslope opening rather than broad active control.
