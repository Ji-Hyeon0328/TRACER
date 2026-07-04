# Phase-B Learned Objective/RAM Offline Policy Score

Offline only: no action selection or low-level controller behavior is changed.

Score = β_v R_v + β_s R_s + β_e R_e + reach_bonus - drift/final/min distance penalties - RAM risk/uncertainty penalties.

| world | rank | policy | score | reach_rate | R_v | R_s | R_e | final_dist | drift | RAM risk | RAM uncertainty |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 1 | tracer_proxy_ppo_argmax_v1 | 4.781 | 1.000 | 9.110 | 2.514 | -0.429 | 0.187 | 0.039 | 0.000 | 0.006 |
| earth | 2 | fixed_trot_solid_fast | 4.639 | 1.000 | 9.121 | 2.414 | -0.742 | 0.243 | 0.097 | 0.000 | 0.006 |
| earth | 3 | fixed_trot_mid | 3.619 | 0.667 | 7.647 | 2.139 | -0.532 | 0.257 | 0.099 | 0.000 | 0.006 |
| earth | 4 | fixed_trot_cautious | 2.576 | 0.333 | 4.882 | 2.004 | -0.333 | 0.373 | 0.091 | 0.000 | 0.006 |
| earth | 5 | fixed_trot_soft_mid_clear | 2.241 | 0.000 | 3.793 | 2.378 | -0.194 | 0.261 | 0.000 | 0.000 | 0.006 |
| stairs_single | 1 | fixed_trot_mid | 4.084 | 1.000 | 9.647 | 1.994 | -0.595 | 0.182 | 0.123 | 0.350 | 0.350 |
| stairs_single | 2 | fixed_trot_cautious | 3.026 | 0.667 | 7.720 | 1.826 | -0.406 | 0.286 | 0.154 | 0.350 | 0.350 |
| stairs_single | 3 | fixed_trot_solid_fast | 2.254 | 0.667 | 6.900 | 1.409 | -0.909 | 0.720 | 0.517 | 0.350 | 0.350 |
| stairs_single | 4 | fixed_trot_soft_mid_clear | 1.757 | 0.333 | 4.591 | 1.645 | -0.369 | 0.511 | 0.205 | 0.350 | 0.350 |
| stairs_single | 5 | tracer_proxy_ppo_argmax_v1 | 1.546 | 0.667 | 7.605 | 0.756 | -1.000 | 1.008 | 0.858 | 0.350 | 0.350 |
| tracer_sponge_firm_flat | 1 | fixed_trot_cautious | 0.445 | 0.333 | 4.610 | 0.877 | -0.475 | 0.851 | 0.557 | 0.878 | 0.534 |
| tracer_sponge_firm_flat | 2 | fixed_trot_soft_mid_clear | -0.174 | 0.000 | 1.326 | 1.402 | -0.377 | 0.763 | 0.291 | 0.878 | 0.534 |
| tracer_sponge_firm_flat | 3 | tracer_proxy_ppo_argmax_v1 | -2.725 | 1.000 | 9.596 | -3.043 | -0.234 | 3.822 | 3.755 | 0.878 | 0.534 |
| tracer_sponge_firm_flat | 4 | fixed_trot_mid | -2.954 | 0.000 | 4.343 | -1.334 | -0.694 | 2.163 | 1.949 | 0.878 | 0.534 |
| tracer_sponge_firm_flat | 5 | fixed_trot_solid_fast | -4.283 | 0.667 | 7.912 | -3.657 | -1.000 | 3.986 | 3.864 | 0.878 | 0.534 |

## Recommended policy per world

- `earth`: `tracer_proxy_ppo_argmax_v1` score=4.781
- `stairs_single`: `fixed_trot_mid` score=4.084
- `tracer_sponge_firm_flat`: `fixed_trot_cautious` score=0.445
