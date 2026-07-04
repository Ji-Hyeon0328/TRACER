# Phase-B β/RAM Candidate Policy v3 Robust-Clean

- Examples: `173`
- Input dim: `22`
- Best epoch: `633`
- Final train MSE: `0.0459`
- Final val MSE: `0.0643`
- Final train pair acc: `1.000`
- Final val pair acc: `1.000`

Robust target:

`mean_score - 0.75 * std_score - 1.5 * mean_risk + 1.0 * reach_rate`

Group statistics are used only as training labels, not as candidate policy inputs.

| world | action | n | pred_policy_score | robust_target | mean_score | std_score | mean_risk | reach_rate | final_dist | hold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | trot_mid | 12 | 10.318 | 8.899 | 7.926 | 0.037 | 0.000 | 1.000 | 0.203 | 1.359 |
| earth | trot_solid_fast | 14 | 6.122 | 5.486 | 6.813 | 2.690 | 0.111 | 0.857 | 0.320 | 1.139 |
| earth | trot_cautious | 5 | 1.798 | 1.882 | 3.711 | 2.066 | 0.320 | 0.200 | 0.172 | 1.426 |
| earth | trot_soft_mid_clear | 5 | 0.283 | 0.917 | 1.871 | 0.171 | 0.550 | 0.000 | 0.253 | 1.326 |
| stairs_single | trot_solid_fast | 16 | 8.673 | 7.669 | 7.966 | 1.572 | 0.038 | 0.938 | 0.486 | 0.697 |
| stairs_single | trot_mid | 10 | 3.580 | 3.483 | 5.692 | 3.275 | 0.235 | 0.600 | 0.503 | 0.643 |
| stairs_single | trot_cautious | 5 | 0.991 | 1.358 | 4.163 | 3.613 | 0.330 | 0.400 | 0.365 | 0.984 |
| stairs_single | trot_soft_mid_clear | 5 | -0.891 | -0.087 | 2.561 | 2.917 | 0.440 | 0.200 | 0.403 | 0.970 |
| tracer_sponge_firm_flat | trot_soft_mid_clear | 17 | 1.192 | 1.044 | 3.426 | 3.036 | 0.579 | 0.765 | 3.656 | -7.445 |
| tracer_sponge_firm_flat | sponge_reach_then_brake | 12 | 0.844 | 1.039 | 3.316 | 2.886 | 0.575 | 0.750 | 3.598 | -7.264 |
| tracer_sponge_firm_flat | sponge_probe_crawlish | 14 | -0.457 | -0.074 | 2.740 | 3.228 | 0.643 | 0.571 | 3.180 | -6.120 |
| tracer_sponge_firm_flat | trot_cautious | 16 | -0.800 | -0.348 | 2.364 | 3.098 | 0.634 | 0.562 | 3.145 | -6.043 |
| tracer_sponge_firm_flat | trot_mid | 17 | -1.273 | -0.663 | 2.128 | 3.133 | 0.647 | 0.529 | 3.406 | -6.778 |
| tracer_sponge_firm_flat | sponge_mid_brake_clear | 5 | -2.011 | -1.253 | 1.539 | 2.816 | 0.720 | 0.400 | 3.397 | -6.747 |
| tracer_sponge_firm_flat | sponge_short_step_stable | 5 | -2.680 | -1.630 | 1.367 | 3.029 | 0.750 | 0.400 | 3.282 | -6.497 |
| tracer_sponge_firm_flat | trot_solid_fast | 10 | -3.337 | -2.034 | 1.116 | 3.263 | 0.735 | 0.400 | 3.426 | -6.825 |
| tracer_sponge_firm_flat | sponge_slow_high_clear | 5 | -4.036 | -2.666 | 0.446 | 2.696 | 0.860 | 0.200 | 2.768 | -5.167 |
