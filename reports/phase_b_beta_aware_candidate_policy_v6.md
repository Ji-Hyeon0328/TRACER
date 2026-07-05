# Phase-B β-aware Candidate Policy v6

- Rows: `173`
- Input dim: `21`
- Best epoch: `1044`
- Best val MSE: `0.002933`

## Group summary

| world::action | n | pred | target_task_beta | target_proxy | reach | final | drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 12 | 1.005 | 1.000 | 7.926 | 1.000 | 0.203 | 0.056 |
| earth::trot_solid_fast | 14 | 0.813 | 0.816 | 6.813 | 0.857 | 0.320 | 0.069 |
| earth::trot_cautious | 5 | 0.289 | 0.291 | 3.711 | 0.200 | 0.172 | 0.000 |
| earth::trot_soft_mid_clear | 5 | -0.003 | 0.000 | 1.871 | 0.000 | 0.253 | 0.000 |
| stairs_single::trot_solid_fast | 16 | 1.003 | 1.000 | 7.966 | 0.938 | 0.486 | 0.382 |
| stairs_single::trot_mid | 10 | 0.566 | 0.566 | 5.692 | 0.600 | 0.503 | 0.265 |
| stairs_single::trot_cautious | 5 | 0.307 | 0.310 | 4.163 | 0.400 | 0.365 | 0.067 |
| stairs_single::trot_soft_mid_clear | 5 | -0.003 | 0.000 | 2.561 | 0.200 | 0.403 | 0.037 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 14 | 0.689 | 0.657 | 2.740 | 0.571 | 3.180 | 1.991 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 12 | 0.675 | 0.655 | 3.316 | 0.750 | 3.598 | 2.637 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 17 | 0.657 | 0.650 | 3.426 | 0.765 | 3.656 | 2.775 |
| tracer_sponge_firm_flat::trot_cautious | 16 | 0.592 | 0.594 | 2.364 | 0.562 | 3.145 | 2.003 |
| tracer_sponge_firm_flat::trot_mid | 17 | 0.467 | 0.475 | 2.128 | 0.529 | 3.406 | 2.092 |
| tracer_sponge_firm_flat::sponge_mid_brake_clear | 5 | 0.386 | 0.401 | 1.539 | 0.400 | 3.397 | 1.417 |
| tracer_sponge_firm_flat::sponge_short_step_stable | 5 | 0.370 | 0.389 | 1.367 | 0.400 | 3.282 | 1.458 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 5 | 0.331 | 0.350 | 0.446 | 0.200 | 2.768 | 0.716 |
| tracer_sponge_firm_flat::trot_solid_fast | 10 | 0.285 | 0.307 | 1.116 | 0.400 | 3.426 | 1.546 |
