# TRACER Phase-I2 RAM Shadow Log Check v0

- csv: `logs/phase_i/i2_ram_shadow_active_20260721_160236/ram_shadow_i2_v0.csv`
- rows: `3036`
- first_t: `1784664157.503585`
- last_t: `1784664461.003616`

## Context counts

| context | rows |
|---|---:|
| downslope | 513 |
| flat | 457 |
| goal_flat | 500 |
| rough | 422 |
| unknown | 729 |
| upslope | 415 |

## Numeric summary

| column | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| rho_slip_pred | 3036 | 0.181896 | 0.372576 | 0.000000 | 1.000000 |
| rho_rough_pred | 3036 | 0.607689 | 0.464598 | 0.000000 | 1.000000 |
| sigma_pred | 3036 | 0.371379 | 0.379433 | 0.000000 | 1.000000 |
| legacy_slip | 2280 | 0.092961 | 0.095351 | 0.000000 | 0.250000 |
| legacy_rough | 2280 | 0.164583 | 0.126053 | 0.050000 | 0.400000 |
| legacy_sigma | 2280 | 0.189079 | 0.124390 | 0.050000 | 0.350000 |
| legacy_l1 | 2280 | 0.963292 | 0.809643 | 0.052132 | 2.900000 |
| observed_features | 3036 | 19.413043 | 6.386231 | 8.000000 | 23.000000 |
| imputed_features | 3036 | 3.586957 | 6.386231 | 0.000000 | 15.000000 |
| x | 2307 | 4.972999 | 2.700942 | -0.002430 | 8.188591 |
| y | 2307 | -0.101120 | 0.108257 | -0.327860 | 0.055151 |
| ref_vx | 2315 | 0.168390 | 0.074165 | 0.025000 | 0.210000 |

## Quick interpretation

- average observed/imputed features: `19.41` / `3.59`
- Most RAM features are observed from runtime/window signals.
- mean L1 vs legacy RAM proxy: `0.963292`
- Large L1 is acceptable in shadow mode because I2 is trained on heuristic teacher seeds, not to copy legacy RAM.
- I2 should remain shadow-only until compared across rollouts and connected to H2/H3 diagnostics.
