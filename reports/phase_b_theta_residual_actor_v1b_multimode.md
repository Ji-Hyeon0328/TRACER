# Phase-B θ Residual Actor v1b Multimode

- Rows: `774`
- Train rows: `620`
- Val rows: `154`
- Input dim: `16`
- Best epoch: `2500`
- Best val MSE: `0.000110`

## Group summary

| group | n | mse_mean |
|---|---:|---:|
| earth::balanced::trot_cautious -> trot_mid | 5 | 0.000005 |
| earth::balanced::trot_mid -> trot_mid | 12 | 0.000005 |
| earth::balanced::trot_soft_mid_clear -> trot_mid | 5 | 0.000003 |
| earth::balanced::trot_solid_fast -> trot_mid | 14 | 0.000003 |
| earth::reach::trot_cautious -> trot_solid_fast | 5 | 0.000483 |
| earth::reach::trot_mid -> trot_solid_fast | 12 | 0.000014 |
| earth::reach::trot_soft_mid_clear -> trot_solid_fast | 5 | 0.000051 |
| earth::reach::trot_solid_fast -> trot_solid_fast | 14 | 0.000007 |
| earth::stability::trot_cautious -> trot_mid | 5 | 0.000004 |
| earth::stability::trot_mid -> trot_mid | 12 | 0.000006 |
| earth::stability::trot_soft_mid_clear -> trot_mid | 5 | 0.000005 |
| earth::stability::trot_solid_fast -> trot_mid | 14 | 0.000003 |
| stairs_single::balanced::trot_cautious -> trot_solid_fast | 5 | 0.000016 |
| stairs_single::balanced::trot_mid -> trot_solid_fast | 10 | 0.000005 |
| stairs_single::balanced::trot_soft_mid_clear -> trot_solid_fast | 5 | 0.000024 |
| stairs_single::balanced::trot_solid_fast -> trot_solid_fast | 16 | 0.000004 |
| stairs_single::reach::trot_cautious -> trot_solid_fast | 5 | 0.000022 |
| stairs_single::reach::trot_mid -> trot_solid_fast | 10 | 0.000006 |
| stairs_single::reach::trot_soft_mid_clear -> trot_solid_fast | 5 | 0.000019 |
| stairs_single::reach::trot_solid_fast -> trot_solid_fast | 16 | 0.000005 |
| stairs_single::stability::trot_cautious -> trot_mid | 5 | 0.000041 |
| stairs_single::stability::trot_mid -> trot_mid | 10 | 0.000032 |
| stairs_single::stability::trot_soft_mid_clear -> trot_mid | 5 | 0.000006 |
| stairs_single::stability::trot_solid_fast -> trot_mid | 16 | 0.000006 |
| tracer_sponge_firm_flat::balanced::sponge_mid_brake_clear -> sponge_v8b_reach_bias | 5 | 0.000043 |
| tracer_sponge_firm_flat::balanced::sponge_probe_crawlish -> sponge_v8b_reach_bias | 19 | 0.000008 |
| tracer_sponge_firm_flat::balanced::sponge_reach_then_brake -> sponge_v8b_reach_bias | 22 | 0.000002 |
| tracer_sponge_firm_flat::balanced::sponge_short_step_stable -> sponge_v8b_reach_bias | 5 | 0.000494 |
| tracer_sponge_firm_flat::balanced::sponge_slow_high_clear -> sponge_v8b_reach_bias | 5 | 0.000019 |
| tracer_sponge_firm_flat::balanced::sponge_v8_anchor_mix_top3 -> sponge_v8b_reach_bias | 10 | 0.000386 |
| tracer_sponge_firm_flat::balanced::sponge_v8b_probe_reach_fast -> sponge_v8b_reach_bias | 5 | 0.000149 |
| tracer_sponge_firm_flat::balanced::sponge_v8b_reach_bias -> sponge_v8b_reach_bias | 5 | 0.000058 |
| tracer_sponge_firm_flat::balanced::sponge_v8b_reach_bias_fast -> sponge_v8b_reach_bias | 10 | 0.000047 |
| tracer_sponge_firm_flat::balanced::sponge_v8b_reach_stabilized -> sponge_v8b_reach_bias | 5 | 0.000076 |
| tracer_sponge_firm_flat::balanced::sponge_v8b_soft_reach -> sponge_v8b_reach_bias | 5 | 0.000191 |
| tracer_sponge_firm_flat::balanced::sponge_v8c_far_fast_early_stop -> sponge_v8b_reach_bias | 5 | 0.000031 |
| tracer_sponge_firm_flat::balanced::sponge_v8c_mid_fast_early_stop -> sponge_v8b_reach_bias | 5 | 0.000357 |
| tracer_sponge_firm_flat::balanced::sponge_v8c_soft_stop -> sponge_v8b_reach_bias | 5 | 0.000077 |
| tracer_sponge_firm_flat::balanced::trot_cautious -> sponge_v8b_reach_bias | 16 | 0.000062 |
| tracer_sponge_firm_flat::balanced::trot_mid -> sponge_v8b_reach_bias | 17 | 0.000396 |
| tracer_sponge_firm_flat::balanced::trot_soft_mid_clear -> sponge_v8b_reach_bias | 32 | 0.000010 |
| tracer_sponge_firm_flat::balanced::trot_solid_fast -> sponge_v8b_reach_bias | 10 | 0.000043 |
| tracer_sponge_firm_flat::reach::sponge_mid_brake_clear -> sponge_v8b_reach_stabilized | 5 | 0.000078 |
| tracer_sponge_firm_flat::reach::sponge_probe_crawlish -> sponge_v8b_reach_stabilized | 19 | 0.000026 |
| tracer_sponge_firm_flat::reach::sponge_reach_then_brake -> sponge_v8b_reach_stabilized | 22 | 0.000044 |
| tracer_sponge_firm_flat::reach::sponge_short_step_stable -> sponge_v8b_reach_stabilized | 5 | 0.000572 |
| tracer_sponge_firm_flat::reach::sponge_slow_high_clear -> sponge_v8b_reach_stabilized | 5 | 0.000066 |
| tracer_sponge_firm_flat::reach::sponge_v8_anchor_mix_top3 -> sponge_v8b_reach_stabilized | 10 | 0.000557 |
| tracer_sponge_firm_flat::reach::sponge_v8b_probe_reach_fast -> sponge_v8b_reach_stabilized | 5 | 0.000173 |
| tracer_sponge_firm_flat::reach::sponge_v8b_reach_bias -> sponge_v8b_reach_stabilized | 5 | 0.000092 |
| tracer_sponge_firm_flat::reach::sponge_v8b_reach_bias_fast -> sponge_v8b_reach_stabilized | 10 | 0.000056 |
| tracer_sponge_firm_flat::reach::sponge_v8b_reach_stabilized -> sponge_v8b_reach_stabilized | 5 | 0.000073 |
| tracer_sponge_firm_flat::reach::sponge_v8b_soft_reach -> sponge_v8b_reach_stabilized | 5 | 0.000436 |
| tracer_sponge_firm_flat::reach::sponge_v8c_far_fast_early_stop -> sponge_v8b_reach_stabilized | 5 | 0.000198 |
| tracer_sponge_firm_flat::reach::sponge_v8c_mid_fast_early_stop -> sponge_v8b_reach_stabilized | 5 | 0.000177 |
| tracer_sponge_firm_flat::reach::sponge_v8c_soft_stop -> sponge_v8b_reach_stabilized | 5 | 0.000627 |
| tracer_sponge_firm_flat::reach::trot_cautious -> sponge_v8b_reach_stabilized | 16 | 0.000162 |
| tracer_sponge_firm_flat::reach::trot_mid -> sponge_v8b_reach_stabilized | 17 | 0.000287 |
| tracer_sponge_firm_flat::reach::trot_soft_mid_clear -> sponge_v8b_reach_stabilized | 32 | 0.000013 |
| tracer_sponge_firm_flat::reach::trot_solid_fast -> sponge_v8b_reach_stabilized | 10 | 0.000075 |
| tracer_sponge_firm_flat::stability::sponge_mid_brake_clear -> sponge_slow_high_clear | 5 | 0.000072 |
| tracer_sponge_firm_flat::stability::sponge_probe_crawlish -> sponge_slow_high_clear | 19 | 0.000048 |
| tracer_sponge_firm_flat::stability::sponge_reach_then_brake -> sponge_slow_high_clear | 22 | 0.000019 |
| tracer_sponge_firm_flat::stability::sponge_short_step_stable -> sponge_slow_high_clear | 5 | 0.001454 |
| tracer_sponge_firm_flat::stability::sponge_slow_high_clear -> sponge_slow_high_clear | 5 | 0.000037 |
| tracer_sponge_firm_flat::stability::sponge_v8_anchor_mix_top3 -> sponge_slow_high_clear | 10 | 0.000268 |
| tracer_sponge_firm_flat::stability::sponge_v8b_probe_reach_fast -> sponge_slow_high_clear | 5 | 0.000054 |
| tracer_sponge_firm_flat::stability::sponge_v8b_reach_bias -> sponge_slow_high_clear | 5 | 0.000043 |
| tracer_sponge_firm_flat::stability::sponge_v8b_reach_bias_fast -> sponge_slow_high_clear | 10 | 0.000187 |
| tracer_sponge_firm_flat::stability::sponge_v8b_reach_stabilized -> sponge_slow_high_clear | 5 | 0.000096 |
| tracer_sponge_firm_flat::stability::sponge_v8b_soft_reach -> sponge_slow_high_clear | 5 | 0.000250 |
| tracer_sponge_firm_flat::stability::sponge_v8c_far_fast_early_stop -> sponge_slow_high_clear | 5 | 0.000013 |
| tracer_sponge_firm_flat::stability::sponge_v8c_mid_fast_early_stop -> sponge_slow_high_clear | 5 | 0.000217 |
| tracer_sponge_firm_flat::stability::sponge_v8c_soft_stop -> sponge_slow_high_clear | 5 | 0.000252 |
| tracer_sponge_firm_flat::stability::trot_cautious -> sponge_slow_high_clear | 16 | 0.000028 |
| tracer_sponge_firm_flat::stability::trot_mid -> sponge_slow_high_clear | 17 | 0.000016 |
| tracer_sponge_firm_flat::stability::trot_soft_mid_clear -> sponge_slow_high_clear | 32 | 0.000013 |
| tracer_sponge_firm_flat::stability::trot_solid_fast -> sponge_slow_high_clear | 10 | 0.000962 |
