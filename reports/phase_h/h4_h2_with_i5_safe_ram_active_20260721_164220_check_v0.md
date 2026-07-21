# TRACER Phase-H2 Objective Selector Shadow Log Check v0

- csv: `logs/phase_h/h4_h2_with_i5_safe_ram_active_20260721_164220/objective_selector_h2_shadow_v0.csv`
- rows: `4082`
- first_t: `1784666540.385919`
- last_t: `1784666948.486196`

## Context counts

| context | rows |
|---|---:|
| downslope | 496 |
| flat | 467 |
| goal_flat | 1448 |
| rough | 435 |
| unknown | 815 |
| upslope | 421 |

## Numeric summary

| column | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| beta_motion | 4082 | 0.835941 | 0.162538 | 0.503165 | 1.000000 |
| beta_stability | 4082 | 0.091291 | 0.089854 | 0.000000 | 0.312496 |
| beta_energy | 4082 | 0.072768 | 0.096950 | 0.000000 | 0.299439 |
| observed_features | 4082 | 41.640862 | 10.745867 | 20.000000 | 47.000000 |
| imputed_features | 4082 | 12.359138 | 10.745867 | 7.000000 | 34.000000 |
| x | 3267 | 5.853583 | 2.696603 | -0.002422 | 8.145709 |
| y | 3267 | 0.060155 | 0.243827 | -0.526359 | 0.325258 |
| ref_vx | 3278 | 0.127077 | 0.090218 | 0.025000 | 0.210000 |
| ram_slip_proxy | 4082 | 0.244965 | 0.091265 | 0.103346 | 0.388346 |
| ram_roughness_proxy | 4082 | 0.363564 | 0.168537 | 0.136920 | 0.648342 |
| ram_sigma | 4082 | 0.324642 | 0.082270 | 0.150479 | 0.439561 |

## Quick interpretation

- average observed/imputed features: `41.64` / `12.36`
- Most features are observed from runtime/window signals.
- H2 beta should not control D6/D7 yet unless compared against rollout behavior.
