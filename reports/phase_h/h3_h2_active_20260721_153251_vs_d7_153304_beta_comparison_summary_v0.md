# TRACER Phase-H3 H2 vs D7 Beta Comparison v0

This compares the Phase-H2 Objective Selector beta shadow against beta outputs logged by the existing D7 objective selector on the same rollout.

- H2 csv: `logs/phase_h/h2_objective_shadow_active_20260721_153251/objective_selector_h2_shadow_v0.csv`
- D7 csv: `logs/phase_d5_shadow_20260721_153304_trial_1/d7_objective_selector_shadow_v0.csv`
- output csv: `datasets/phase_h/h3_h2_active_20260721_153251_vs_d7_153304_beta_comparison_v0.csv`
- matched records: `8516`
- skipped H2 rows: `1246`
- max dt: `0.35` seconds
- detected D7 beta groups: `actual_beta, pred_beta, prior_beta, raw_beta`

## Overall comparison by D7 beta group

| D7 group | records | mean L1 | std L1 | max L1 | mean |H2-D7| dt | H2 beta mean | D7 beta mean |
|---|---:|---:|---:|---:|---:|---|---|
| actual_beta | 2126 | 0.740735 | 0.389789 | 1.365727 | 0.039455 | (0.6144, 0.1923, 0.1933) | (0.3015, 0.4628, 0.2357) |
| pred_beta | 2130 | 0.704769 | 0.337814 | 1.287125 | 0.039625 | (0.6151, 0.1919, 0.1929) | (0.3434, 0.4212, 0.2355) |
| prior_beta | 2130 | 0.741540 | 0.389862 | 1.365727 | 0.039625 | (0.6151, 0.1919, 0.1929) | (0.3016, 0.4626, 0.2358) |
| raw_beta | 2130 | 0.698535 | 0.324208 | 1.268166 | 0.039625 | (0.6151, 0.1919, 0.1929) | (0.3538, 0.4108, 0.2354) |

## Known-context-only comparison by D7 beta group

| D7 group | records | mean L1 | std L1 | max L1 | mean dt | H2 beta mean | D7 beta mean |
|---|---:|---:|---:|---:|---:|---|---|
| actual_beta | 2126 | 0.740735 | 0.389789 | 1.365727 | 0.039455 | (0.6144, 0.1923, 0.1933) | (0.3015, 0.4628, 0.2357) |
| pred_beta | 2130 | 0.704769 | 0.337814 | 1.287125 | 0.039625 | (0.6151, 0.1919, 0.1929) | (0.3434, 0.4212, 0.2355) |
| prior_beta | 2130 | 0.741540 | 0.389862 | 1.365727 | 0.039625 | (0.6151, 0.1919, 0.1929) | (0.3016, 0.4626, 0.2358) |
| raw_beta | 2130 | 0.698535 | 0.324208 | 1.268166 | 0.039625 | (0.6151, 0.1919, 0.1929) | (0.3538, 0.4108, 0.2354) |

## Known-context-only comparison by terrain context

| D7 group | context | records | mean L1 | max L1 | H2 beta mean | D7 beta mean |
|---|---|---:|---:|---:|---|---|
| actual_beta | downslope | 486 | 0.978959 | 1.355969 | (0.7380, 0.2163, 0.0457) | (0.2500, 0.5998, 0.1502) |
| actual_beta | flat | 423 | 0.286245 | 1.100000 | (0.4063, 0.3302, 0.2635) | (0.4500, 0.2500, 0.3000) |
| actual_beta | goal_flat | 364 | 0.570661 | 0.641250 | (0.4855, 0.2689, 0.2456) | (0.2001, 0.5003, 0.2996) |
| actual_beta | rough | 427 | 1.302126 | 1.365727 | (0.8850, 0.0049, 0.1101) | (0.2505, 0.5493, 0.2002) |
| actual_beta | upslope | 426 | 0.502861 | 0.590060 | (0.5191, 0.1503, 0.3307) | (0.3505, 0.3993, 0.2502) |
| pred_beta | downslope | 486 | 0.913045 | 1.280461 | (0.7380, 0.2163, 0.0457) | (0.2815, 0.5451, 0.1733) |
| pred_beta | flat | 427 | 0.357526 | 0.955646 | (0.4119, 0.3271, 0.2610) | (0.5140, 0.2068, 0.2792) |
| pred_beta | goal_flat | 364 | 0.597279 | 0.657230 | (0.4855, 0.2689, 0.2456) | (0.1916, 0.5244, 0.2840) |
| pred_beta | rough | 427 | 1.196927 | 1.287125 | (0.8850, 0.0049, 0.1101) | (0.3022, 0.4869, 0.2108) |
| pred_beta | upslope | 426 | 0.413750 | 0.502025 | (0.5191, 0.1503, 0.3307) | (0.4138, 0.3405, 0.2457) |
| prior_beta | downslope | 486 | 0.979047 | 1.355969 | (0.7380, 0.2163, 0.0457) | (0.2500, 0.5999, 0.1501) |
| prior_beta | flat | 427 | 0.293868 | 1.100000 | (0.4119, 0.3271, 0.2610) | (0.4500, 0.2500, 0.3000) |
| prior_beta | goal_flat | 364 | 0.570936 | 0.660470 | (0.4855, 0.2689, 0.2456) | (0.2000, 0.5000, 0.3000) |
| prior_beta | rough | 427 | 1.302828 | 1.365727 | (0.8850, 0.0049, 0.1101) | (0.2502, 0.5496, 0.2001) |
| prior_beta | upslope | 426 | 0.502473 | 0.590060 | (0.5191, 0.1503, 0.3307) | (0.3502, 0.3996, 0.2501) |
| raw_beta | downslope | 486 | 0.897144 | 1.261584 | (0.7380, 0.2163, 0.0457) | (0.2894, 0.5315, 0.1791) |
| raw_beta | flat | 427 | 0.380419 | 0.919557 | (0.4119, 0.3271, 0.2610) | (0.5300, 0.1960, 0.2740) |
| raw_beta | goal_flat | 364 | 0.608936 | 0.688931 | (0.4855, 0.2689, 0.2456) | (0.1895, 0.5305, 0.2800) |
| raw_beta | rough | 427 | 1.170457 | 1.268166 | (0.8850, 0.0049, 0.1101) | (0.3152, 0.4713, 0.2135) |
| raw_beta | upslope | 426 | 0.394345 | 0.480214 | (0.5191, 0.1503, 0.3307) | (0.4297, 0.3257, 0.2446) |

## Safe interpretation

- This is a shadow comparison only. H2 beta is not used for active control here.
- Large H2-D7 L1 does not automatically mean H2 is wrong, because H2 is trained toward robust true-metric teacher seeds while D7 is the earlier runtime-aligned selector.
- If H2 is much more motion-heavy or frequently saturates at a simplex corner, keep it shadow-only and improve RAM/context coverage before active deployment.
- If terrain transitions are visible and L1 is moderate, H2 can remain the candidate Objective Selector pretraining model for the next RAM/RL stages.
