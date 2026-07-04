# Phase-B β/RAM Robust Candidate Policy v2

- Examples: `60`
- Input dim: `26`
- Best epoch: `237`
- Final train MSE: `0.0134`
- Final val MSE: `0.0190`
- Final train pair acc: `1.000`
- Final val pair acc: `1.000`

Robust target:

`mean_score - 0.75 * std_score - 1.5 * mean_risk + 1.0 * reach_rate`

| world | action | n | pred_policy_score | robust_target | mean_score | std_score | mean_risk | reach_rate | final_dist | hold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | trot_mid | 5 | 9.566 | 8.903 | 7.926 | 0.031 | 0.000 | 1.000 | 0.203 | 1.366 |
| earth | trot_solid_fast | 5 | 8.854 | 8.884 | 7.910 | 0.034 | 0.000 | 1.000 | 0.232 | 1.327 |
| earth | trot_cautious | 5 | 1.632 | 1.882 | 3.711 | 2.066 | 0.320 | 0.200 | 0.172 | 1.426 |
| earth | trot_soft_mid_clear | 5 | 0.374 | 0.917 | 1.871 | 0.171 | 0.550 | 0.000 | 0.253 | 1.326 |
| stairs_single | trot_solid_fast | 5 | 9.485 | 8.795 | 8.474 | 0.864 | 0.020 | 1.000 | 0.537 | 0.557 |
| stairs_single | trot_mid | 5 | 3.476 | 3.365 | 5.736 | 3.581 | 0.190 | 0.600 | 0.397 | 0.900 |
| stairs_single | trot_cautious | 5 | 0.897 | 1.358 | 4.163 | 3.613 | 0.330 | 0.400 | 0.365 | 0.984 |
| stairs_single | trot_soft_mid_clear | 5 | -0.883 | -0.087 | 2.561 | 2.917 | 0.440 | 0.200 | 0.403 | 0.970 |
| tracer_sponge_firm_flat | trot_mid | 5 | 1.551 | 1.312 | 3.348 | 2.354 | 0.580 | 0.600 | 3.090 | -5.961 |
| tracer_sponge_firm_flat | trot_soft_mid_clear | 5 | -0.522 | -0.640 | 2.230 | 3.287 | 0.670 | 0.600 | 3.435 | -6.917 |
| tracer_sponge_firm_flat | trot_cautious | 5 | -1.973 | -1.653 | 1.588 | 3.295 | 0.780 | 0.400 | 3.149 | -6.090 |
| tracer_sponge_firm_flat | trot_solid_fast | 5 | -4.285 | -3.509 | -0.436 | 2.704 | 0.830 | 0.200 | 3.533 | -7.120 |

This robust scorer is intended to reduce soft-terrain over-selection caused by high-variance mean-score targets.
