# Phase-B θ Residual Actor v0

- Rows: `258`
- Train rows: `207`
- Val rows: `51`
- Input dim: `13`
- Best epoch: `2400`
- Best val MSE: `0.000118`

## Group summary

| group | n | mse_mean |
|---|---:|---:|
| earth::trot_cautious -> trot_mid | 5 | 0.000050 |
| earth::trot_mid -> trot_mid | 12 | 0.000011 |
| earth::trot_soft_mid_clear -> trot_mid | 5 | 0.000006 |
| earth::trot_solid_fast -> trot_mid | 14 | 0.000001 |
| stairs_single::trot_cautious -> trot_solid_fast | 5 | 0.000019 |
| stairs_single::trot_mid -> trot_solid_fast | 10 | 0.000032 |
| stairs_single::trot_soft_mid_clear -> trot_solid_fast | 5 | 0.000005 |
| stairs_single::trot_solid_fast -> trot_solid_fast | 16 | 0.000020 |
| tracer_sponge_firm_flat::sponge_mid_brake_clear -> sponge_v8c_far_fast_early_stop | 5 | 0.000078 |
| tracer_sponge_firm_flat::sponge_probe_crawlish -> sponge_v8c_far_fast_early_stop | 19 | 0.000011 |
| tracer_sponge_firm_flat::sponge_reach_then_brake -> sponge_v8c_far_fast_early_stop | 22 | 0.000016 |
| tracer_sponge_firm_flat::sponge_short_step_stable -> sponge_v8c_far_fast_early_stop | 5 | 0.000301 |
| tracer_sponge_firm_flat::sponge_slow_high_clear -> sponge_v8c_far_fast_early_stop | 5 | 0.000188 |
| tracer_sponge_firm_flat::sponge_v8_anchor_mix_top3 -> sponge_v8c_far_fast_early_stop | 10 | 0.000623 |
| tracer_sponge_firm_flat::sponge_v8b_probe_reach_fast -> sponge_v8c_far_fast_early_stop | 5 | 0.000156 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias -> sponge_v8c_far_fast_early_stop | 5 | 0.000210 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias_fast -> sponge_v8c_far_fast_early_stop | 10 | 0.000077 |
| tracer_sponge_firm_flat::sponge_v8b_reach_stabilized -> sponge_v8c_far_fast_early_stop | 5 | 0.000471 |
| tracer_sponge_firm_flat::sponge_v8b_soft_reach -> sponge_v8c_far_fast_early_stop | 5 | 0.000296 |
| tracer_sponge_firm_flat::sponge_v8c_far_fast_early_stop -> sponge_v8c_far_fast_early_stop | 5 | 0.000022 |
| tracer_sponge_firm_flat::sponge_v8c_mid_fast_early_stop -> sponge_v8c_far_fast_early_stop | 5 | 0.000155 |
| tracer_sponge_firm_flat::sponge_v8c_soft_stop -> sponge_v8c_far_fast_early_stop | 5 | 0.000078 |
| tracer_sponge_firm_flat::trot_cautious -> sponge_v8c_far_fast_early_stop | 16 | 0.000301 |
| tracer_sponge_firm_flat::trot_mid -> sponge_v8c_far_fast_early_stop | 17 | 0.000158 |
| tracer_sponge_firm_flat::trot_soft_mid_clear -> sponge_v8c_far_fast_early_stop | 32 | 0.000019 |
| tracer_sponge_firm_flat::trot_solid_fast -> sponge_v8c_far_fast_early_stop | 10 | 0.000006 |
