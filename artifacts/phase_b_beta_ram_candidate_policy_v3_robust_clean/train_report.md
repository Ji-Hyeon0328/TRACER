# Phase-B β/RAM Candidate Policy v3 Robust-Clean

- Examples: `105`
- Input dim: `22`
- Best epoch: `510`
- Final train MSE: `0.0144`
- Final val MSE: `0.0180`
- Final train pair acc: `1.000`
- Final val pair acc: `1.000`

Robust target:

`mean_score - 0.75 * std_score - 1.5 * mean_risk + 1.0 * reach_rate`

Group statistics are used only as training labels, not as candidate policy inputs.

| world | action | n | pred_policy_score | robust_target | mean_score | std_score | mean_risk | reach_rate | final_dist | hold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | trot_mid | 5 | 9.460 | 8.903 | 7.926 | 0.031 | 0.000 | 1.000 | 0.203 | 1.366 |
| earth | trot_solid_fast | 5 | 9.034 | 8.884 | 7.910 | 0.034 | 0.000 | 1.000 | 0.232 | 1.327 |
| earth | trot_cautious | 5 | 2.101 | 1.882 | 3.711 | 2.066 | 0.320 | 0.200 | 0.172 | 1.426 |
| earth | trot_soft_mid_clear | 5 | 0.644 | 0.917 | 1.871 | 0.171 | 0.550 | 0.000 | 0.253 | 1.326 |
| stairs_single | trot_solid_fast | 5 | 9.280 | 8.795 | 8.474 | 0.864 | 0.020 | 1.000 | 0.537 | 0.557 |
| stairs_single | trot_mid | 5 | 3.703 | 3.365 | 5.736 | 3.581 | 0.190 | 0.600 | 0.397 | 0.900 |
| stairs_single | trot_cautious | 5 | 1.041 | 1.358 | 4.163 | 3.613 | 0.330 | 0.400 | 0.365 | 0.984 |
| stairs_single | trot_soft_mid_clear | 5 | -0.532 | -0.087 | 2.561 | 2.917 | 0.440 | 0.200 | 0.403 | 0.970 |
| tracer_sponge_firm_flat | sponge_reach_then_brake | 5 | 2.362 | 1.951 | 3.475 | 2.039 | 0.530 | 0.800 | 3.749 | -7.663 |
| tracer_sponge_firm_flat | trot_mid | 10 | 1.803 | 1.600 | 3.538 | 2.407 | 0.555 | 0.700 | 3.541 | -7.090 |
| tracer_sponge_firm_flat | sponge_probe_crawlish | 5 | 1.745 | 1.558 | 3.825 | 2.969 | 0.560 | 0.800 | 3.549 | -7.016 |
| tracer_sponge_firm_flat | trot_soft_mid_clear | 10 | 0.328 | 0.340 | 2.748 | 2.915 | 0.615 | 0.700 | 3.801 | -7.835 |
| tracer_sponge_firm_flat | trot_cautious | 10 | -0.175 | 0.052 | 2.618 | 2.971 | 0.625 | 0.600 | 3.217 | -6.221 |
| tracer_sponge_firm_flat | sponge_mid_brake_clear | 5 | -1.556 | -1.253 | 1.539 | 2.816 | 0.720 | 0.400 | 3.397 | -6.747 |
| tracer_sponge_firm_flat | sponge_short_step_stable | 5 | -2.083 | -1.630 | 1.367 | 3.029 | 0.750 | 0.400 | 3.282 | -6.497 |
| tracer_sponge_firm_flat | trot_solid_fast | 10 | -2.830 | -2.034 | 1.116 | 3.263 | 0.735 | 0.400 | 3.426 | -6.825 |
| tracer_sponge_firm_flat | sponge_slow_high_clear | 5 | -3.400 | -2.666 | 0.446 | 2.696 | 0.860 | 0.200 | 2.768 | -5.167 |
