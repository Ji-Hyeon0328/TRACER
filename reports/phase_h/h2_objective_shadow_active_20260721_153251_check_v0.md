# TRACER Phase-H2 Objective Selector Shadow Log Check v0

- csv: `logs/phase_h/h2_objective_shadow_active_20260721_153251/objective_selector_h2_shadow_v0.csv`
- rows: `3376`
- first_t: `1784662371.621859`
- last_t: `1784662709.121828`

## Context counts

| context | rows |
|---|---:|
| downslope | 486 |
| flat | 460 |
| goal_flat | 853 |
| rough | 427 |
| unknown | 724 |
| upslope | 426 |

## Numeric summary

| column | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| beta_motion | 3376 | 0.682997 | 0.237545 | 0.333327 | 1.000000 |
| beta_stability | 3376 | 0.154909 | 0.135934 | 0.000000 | 0.452758 |
| beta_energy | 3376 | 0.162094 | 0.133703 | 0.000000 | 0.383371 |
| observed_features | 3376 | 38.532287 | 15.949960 | 8.000000 | 47.000000 |
| imputed_features | 3376 | 15.467713 | 15.949960 | 7.000000 | 46.000000 |
| x | 2652 | 5.336015 | 2.743323 | -0.001887 | 8.132015 |
| y | 2652 | -0.013972 | 0.105877 | -0.197270 | 0.138400 |
| ref_vx | 2659 | 0.149283 | 0.084574 | 0.025000 | 0.210000 |
| ram_slip_proxy | 2616 | 0.078956 | 0.092811 | 0.000000 | 0.250000 |
| ram_roughness_proxy | 2616 | 0.150096 | 0.124475 | 0.050000 | 0.400000 |
| ram_sigma | 2616 | 0.169744 | 0.124912 | 0.050000 | 0.350000 |

## Quick interpretation

- average observed/imputed features: `38.53` / `15.47`
- Most features are observed from runtime/window signals.
- H2 beta should not control D6/D7 yet unless compared against rollout behavior.
