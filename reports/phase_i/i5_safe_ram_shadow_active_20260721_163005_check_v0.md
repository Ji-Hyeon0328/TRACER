# TRACER Phase-I5 Safe RAM Shadow Log Check v0

- csv: `logs/phase_i/i5_safe_ram_shadow_active_20260721_163005/safe_ram_shadow_i5_v0.csv`
- rows: `5331`
- first_t: `1784665806.667434`
- last_t: `1784666339.667483`

## Context counts

| context | rows |
|---|---:|
| downslope | 514 |
| flat | 469 |
| goal_flat | 590 |
| rough | 429 |
| unknown | 2904 |
| upslope | 425 |

## Numeric summary

| column | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| safe_slip | 5331 | 0.257574 | 0.082737 | 0.103346 | 0.388346 |
| safe_rough | 5331 | 0.482510 | 0.198749 | 0.136920 | 0.648342 |
| safe_sigma | 5331 | 0.327657 | 0.071605 | 0.150479 | 0.436192 |
| learned_slip | 5331 | 0.108273 | 0.297753 | 0.000000 | 1.000000 |
| learned_rough | 5331 | 0.744582 | 0.427770 | 0.000000 | 1.000000 |
| learned_sigma | 5331 | 0.215094 | 0.344080 | 0.000000 | 1.000000 |
| damped_slip | 5331 | 0.333170 | 0.104213 | 0.295275 | 0.645275 |
| damped_rough | 5331 | 0.558946 | 0.149719 | 0.298342 | 0.648342 |
| damped_sigma | 5331 | 0.412365 | 0.120428 | 0.337082 | 0.687082 |
| legacy_slip | 2140 | 0.099953 | 0.094930 | 0.000000 | 0.250000 |
| legacy_rough | 2140 | 0.173972 | 0.126722 | 0.050000 | 0.400000 |
| legacy_sigma | 2140 | 0.199977 | 0.122594 | 0.050000 | 0.350000 |
| learned_l1_legacy | 2140 | 1.007122 | 0.827850 | 0.050410 | 2.900000 |
| damped_l1_legacy | 2140 | 0.911489 | 0.312285 | 0.368849 | 1.880699 |
| safe_l1_legacy | 2140 | 0.319021 | 0.109300 | 0.129097 | 0.658245 |
| used_legacy | 5331 | 0.401426 | 0.490187 | 0.000000 | 1.000000 |
| x | 2427 | 5.037937 | 2.705636 | -0.001944 | 8.103846 |
| y | 2427 | -0.283161 | 0.295656 | -0.660502 | 0.111324 |

## Saturation

| output | slip | rough | sigma |
|---|---:|---:|---:|
| learned | 0.921028 | 0.941287 | 0.757081 |
| damped | 0.000000 | 0.000000 | 0.000000 |
| safe | 0.000000 | 0.000000 | 0.000000 |

## Quick interpretation

- learned L1 vs legacy: `1.007122`
- safe L1 vs legacy: `0.319021`
- I5 safe postprocess reduces distance from legacy RAM while preserving learned RAM direction.
- I5 is still a shadow output. Do not replace `/tracer/ram_mismatch` yet.
