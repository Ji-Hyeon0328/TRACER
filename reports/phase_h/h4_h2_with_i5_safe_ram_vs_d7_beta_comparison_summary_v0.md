# TRACER Phase-H3 H2 vs D7 Beta Comparison v0

This compares the Phase-H2 Objective Selector beta shadow against beta outputs logged by the existing D7 objective selector on the same rollout.

- H2 csv: `logs/phase_h/h4_h2_with_i5_safe_ram_active_20260721_164220/objective_selector_h2_shadow_v0.csv`
- D7 csv: `logs/phase_d5_shadow_20260721_164242_trial_1/d7_objective_selector_shadow_v0.csv`
- output csv: `datasets/phase_h/h4_h2_with_i5_safe_ram_vs_d7_beta_comparison_v0.csv`
- matched records: `8520`
- skipped H2 rows: `1952`
- max dt: `0.35` seconds
- detected D7 beta groups: `actual_beta, pred_beta, prior_beta, raw_beta`

## Overall comparison by D7 beta group

| D7 group | records | mean L1 | std L1 | max L1 | mean |H2-D7| dt | H2 beta mean | D7 beta mean |
|---|---:|---:|---:|---:|---:|---|---|
| actual_beta | 2130 | 0.930234 | 0.449317 | 1.500000 | 0.012056 | (0.7568, 0.1230, 0.1202) | (0.3032, 0.4621, 0.2347) |
| pred_beta | 2130 | 0.881898 | 0.452543 | 1.483625 | 0.012056 | (0.7568, 0.1230, 0.1202) | (0.3365, 0.4336, 0.2299) |
| prior_beta | 2130 | 0.930422 | 0.449325 | 1.500000 | 0.012056 | (0.7568, 0.1230, 0.1202) | (0.3031, 0.4622, 0.2347) |
| raw_beta | 2130 | 0.870837 | 0.452831 | 1.479532 | 0.012056 | (0.7568, 0.1230, 0.1202) | (0.3448, 0.4265, 0.2287) |

## Known-context-only comparison by D7 beta group

| D7 group | records | mean L1 | std L1 | max L1 | mean dt | H2 beta mean | D7 beta mean |
|---|---:|---:|---:|---:|---:|---|---|
| actual_beta | 2130 | 0.930234 | 0.449317 | 1.500000 | 0.012056 | (0.7568, 0.1230, 0.1202) | (0.3032, 0.4621, 0.2347) |
| pred_beta | 2130 | 0.881898 | 0.452543 | 1.483625 | 0.012056 | (0.7568, 0.1230, 0.1202) | (0.3365, 0.4336, 0.2299) |
| prior_beta | 2130 | 0.930422 | 0.449325 | 1.500000 | 0.012056 | (0.7568, 0.1230, 0.1202) | (0.3031, 0.4622, 0.2347) |
| raw_beta | 2130 | 0.870837 | 0.452831 | 1.479532 | 0.012056 | (0.7568, 0.1230, 0.1202) | (0.3448, 0.4265, 0.2287) |

## Known-context-only comparison by terrain context

| D7 group | context | records | mean L1 | max L1 | H2 beta mean | D7 beta mean |
|---|---|---:|---:|---:|---|---|
| actual_beta | downslope | 496 | 1.317411 | 1.461090 | (0.9087, 0.0879, 0.0034) | (0.2500, 0.6000, 0.1500) |
| actual_beta | flat | 440 | 0.330717 | 1.100000 | (0.5892, 0.2479, 0.1628) | (0.4500, 0.2500, 0.3000) |
| actual_beta | goal_flat | 338 | 0.913303 | 1.276175 | (0.6568, 0.1939, 0.1493) | (0.2001, 0.5003, 0.2996) |
| actual_beta | rough | 435 | 1.424746 | 1.500000 | (0.9557, 0.0016, 0.0427) | (0.2500, 0.5500, 0.2000) |
| actual_beta | upslope | 421 | 0.603295 | 0.674260 | (0.6278, 0.1022, 0.2700) | (0.3500, 0.4000, 0.2500) |
| pred_beta | downslope | 496 | 1.251220 | 1.405948 | (0.9087, 0.0879, 0.0034) | (0.2831, 0.5414, 0.1755) |
| pred_beta | flat | 440 | 0.281409 | 0.955439 | (0.5892, 0.2479, 0.1628) | (0.5121, 0.2085, 0.2793) |
| pred_beta | goal_flat | 338 | 0.968628 | 1.335528 | (0.6568, 0.1939, 0.1493) | (0.1725, 0.5539, 0.2736) |
| pred_beta | rough | 435 | 1.368024 | 1.483625 | (0.9557, 0.0016, 0.0427) | (0.2787, 0.5257, 0.1956) |
| pred_beta | upslope | 421 | 0.502449 | 0.863104 | (0.6278, 0.1022, 0.2700) | (0.4073, 0.3502, 0.2425) |
| prior_beta | downslope | 496 | 1.317411 | 1.461090 | (0.9087, 0.0879, 0.0034) | (0.2500, 0.6000, 0.1500) |
| prior_beta | flat | 440 | 0.330717 | 1.100000 | (0.5892, 0.2479, 0.1628) | (0.4500, 0.2500, 0.3000) |
| prior_beta | goal_flat | 338 | 0.913599 | 1.376175 | (0.6568, 0.1939, 0.1493) | (0.2000, 0.5000, 0.3000) |
| prior_beta | rough | 435 | 1.424746 | 1.500000 | (0.9557, 0.0016, 0.0427) | (0.2500, 0.5501, 0.1999) |
| prior_beta | upslope | 421 | 0.604007 | 0.974260 | (0.6278, 0.1022, 0.2700) | (0.3498, 0.4004, 0.2499) |
| raw_beta | downslope | 496 | 1.234672 | 1.392162 | (0.9087, 0.0879, 0.0034) | (0.2914, 0.5267, 0.1819) |
| raw_beta | flat | 440 | 0.274239 | 0.919298 | (0.5892, 0.2479, 0.1628) | (0.5276, 0.1982, 0.2742) |
| raw_beta | goal_flat | 338 | 0.982385 | 1.325366 | (0.6568, 0.1939, 0.1493) | (0.1656, 0.5674, 0.2670) |
| raw_beta | rough | 435 | 1.353846 | 1.479532 | (0.9557, 0.0016, 0.0427) | (0.2858, 0.5196, 0.1946) |
| raw_beta | upslope | 421 | 0.477083 | 0.835315 | (0.6278, 0.1022, 0.2700) | (0.4216, 0.3377, 0.2407) |

## Safe interpretation

- This is a shadow comparison only. H2 beta is not used for active control here.
- Large H2-D7 L1 does not automatically mean H2 is wrong, because H2 is trained toward robust true-metric teacher seeds while D7 is the earlier runtime-aligned selector.
- If H2 is much more motion-heavy or frequently saturates at a simplex corner, keep it shadow-only and improve RAM/context coverage before active deployment.
- If terrain transitions are visible and L1 is moderate, H2 can remain the candidate Objective Selector pretraining model for the next RAM/RL stages.
