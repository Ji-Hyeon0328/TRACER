# Phase-B β-aware Candidate Policy v7 Deployable

- Rows: `173`
- Input dim: `18`
- Best epoch: `1729`
- Best val MSE: `0.003722`

## Group summary

| world::action | n | pred | target_task_beta | proxy | reach | final | drift | beta_v | beta_s | beta_e |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 12 | 1.007 | 1.000 | 7.926 | 1.000 | 0.203 | 0.056 | 0.600 | 0.231 | 0.169 |
| earth::trot_solid_fast | 14 | 0.812 | 0.816 | 6.813 | 0.857 | 0.320 | 0.069 | 0.600 | 0.231 | 0.169 |
| earth::trot_cautious | 5 | 0.288 | 0.291 | 3.711 | 0.200 | 0.172 | 0.000 | 0.600 | 0.231 | 0.169 |
| earth::trot_soft_mid_clear | 5 | -0.007 | 0.000 | 1.871 | 0.000 | 0.253 | 0.000 | 0.600 | 0.231 | 0.169 |
| stairs_single::trot_solid_fast | 16 | 1.001 | 1.000 | 7.966 | 0.938 | 0.486 | 0.382 | 0.465 | 0.399 | 0.136 |
| stairs_single::trot_mid | 10 | 0.564 | 0.566 | 5.692 | 0.600 | 0.503 | 0.265 | 0.465 | 0.399 | 0.136 |
| stairs_single::trot_cautious | 5 | 0.304 | 0.310 | 4.163 | 0.400 | 0.365 | 0.067 | 0.465 | 0.399 | 0.136 |
| stairs_single::trot_soft_mid_clear | 5 | -0.006 | 0.000 | 2.561 | 0.200 | 0.403 | 0.037 | 0.465 | 0.399 | 0.136 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 14 | 0.688 | 0.657 | 2.740 | 0.571 | 3.180 | 1.991 | 0.142 | 0.795 | 0.063 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 12 | 0.677 | 0.655 | 3.316 | 0.750 | 3.598 | 2.637 | 0.142 | 0.795 | 0.063 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 17 | 0.660 | 0.650 | 3.426 | 0.765 | 3.656 | 2.775 | 0.142 | 0.795 | 0.063 |
| tracer_sponge_firm_flat::trot_cautious | 16 | 0.585 | 0.594 | 2.364 | 0.562 | 3.145 | 2.003 | 0.142 | 0.795 | 0.063 |
| tracer_sponge_firm_flat::trot_mid | 17 | 0.461 | 0.475 | 2.128 | 0.529 | 3.406 | 2.092 | 0.142 | 0.795 | 0.063 |
| tracer_sponge_firm_flat::sponge_mid_brake_clear | 5 | 0.385 | 0.401 | 1.539 | 0.400 | 3.397 | 1.417 | 0.142 | 0.795 | 0.063 |
| tracer_sponge_firm_flat::sponge_short_step_stable | 5 | 0.367 | 0.389 | 1.367 | 0.400 | 3.282 | 1.458 | 0.142 | 0.795 | 0.063 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 5 | 0.326 | 0.350 | 0.446 | 0.200 | 2.768 | 0.716 | 0.142 | 0.795 | 0.063 |
| tracer_sponge_firm_flat::trot_solid_fast | 10 | 0.282 | 0.307 | 1.116 | 0.400 | 3.426 | 1.546 | 0.142 | 0.795 | 0.063 |
