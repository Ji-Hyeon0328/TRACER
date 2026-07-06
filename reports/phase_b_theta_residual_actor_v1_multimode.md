# Phase-B θ Residual Actor v1 Multimode

- Rows: `774`
- Train rows: `620`
- Val rows: `154`
- Input dim: `16`
- Best epoch: `2500`
- Best val MSE: `0.000157`

## Group summary

| group | n | mse_mean |
|---|---:|---:|
| earth::balanced::trot_cautious -> trot_mid | 5 | 0.000016 |
| earth::balanced::trot_mid -> trot_mid | 12 | 0.000003 |
| earth::balanced::trot_soft_mid_clear -> trot_mid | 5 | 0.000007 |
| earth::balanced::trot_solid_fast -> trot_mid | 14 | 0.000002 |
| earth::reach::trot_cautious -> trot_solid_fast | 5 | 0.000655 |
| earth::reach::trot_mid -> trot_solid_fast | 12 | 0.000011 |
| earth::reach::trot_soft_mid_clear -> trot_solid_fast | 5 | 0.000074 |
| earth::reach::trot_solid_fast -> trot_solid_fast | 14 | 0.000002 |
| earth::stability::trot_cautious -> trot_mid | 5 | 0.000016 |
| earth::stability::trot_mid -> trot_mid | 12 | 0.000004 |
| earth::stability::trot_soft_mid_clear -> trot_mid | 5 | 0.000033 |
| earth::stability::trot_solid_fast -> trot_mid | 14 | 0.000004 |
| stairs_single::balanced::trot_cautious -> trot_solid_fast | 5 | 0.000054 |
| stairs_single::balanced::trot_mid -> trot_solid_fast | 10 | 0.000006 |
| stairs_single::balanced::trot_soft_mid_clear -> trot_solid_fast | 5 | 0.000101 |
| stairs_single::balanced::trot_solid_fast -> trot_solid_fast | 16 | 0.000006 |
| stairs_single::reach::trot_cautious -> trot_solid_fast | 5 | 0.000049 |
| stairs_single::reach::trot_mid -> trot_solid_fast | 10 | 0.000010 |
| stairs_single::reach::trot_soft_mid_clear -> trot_solid_fast | 5 | 0.000022 |
| stairs_single::reach::trot_solid_fast -> trot_solid_fast | 16 | 0.000005 |
| stairs_single::stability::trot_cautious -> trot_mid | 5 | 0.000192 |
| stairs_single::stability::trot_mid -> trot_mid | 10 | 0.000074 |
| stairs_single::stability::trot_soft_mid_clear -> trot_mid | 5 | 0.000094 |
| stairs_single::stability::trot_solid_fast -> trot_mid | 16 | 0.000007 |
| tracer_sponge_firm_flat::balanced::sponge_mid_brake_clear -> sponge_v8c_far_fast_early_stop | 5 | 0.000187 |
| tracer_sponge_firm_flat::balanced::sponge_probe_crawlish -> sponge_v8c_far_fast_early_stop | 19 | 0.000053 |
| tracer_sponge_firm_flat::balanced::sponge_reach_then_brake -> sponge_v8c_far_fast_early_stop | 22 | 0.000030 |
| tracer_sponge_firm_flat::balanced::sponge_short_step_stable -> sponge_v8c_far_fast_early_stop | 5 | 0.000907 |
| tracer_sponge_firm_flat::balanced::sponge_slow_high_clear -> sponge_v8c_far_fast_early_stop | 5 | 0.000240 |
| tracer_sponge_firm_flat::balanced::sponge_v8_anchor_mix_top3 -> sponge_v8c_far_fast_early_stop | 10 | 0.000658 |
| tracer_sponge_firm_flat::balanced::sponge_v8b_probe_reach_fast -> sponge_v8c_far_fast_early_stop | 5 | 0.000387 |
| tracer_sponge_firm_flat::balanced::sponge_v8b_reach_bias -> sponge_v8c_far_fast_early_stop | 5 | 0.000182 |
| tracer_sponge_firm_flat::balanced::sponge_v8b_reach_bias_fast -> sponge_v8c_far_fast_early_stop | 10 | 0.000207 |
| tracer_sponge_firm_flat::balanced::sponge_v8b_reach_stabilized -> sponge_v8c_far_fast_early_stop | 5 | 0.000152 |
| tracer_sponge_firm_flat::balanced::sponge_v8b_soft_reach -> sponge_v8c_far_fast_early_stop | 5 | 0.000583 |
| tracer_sponge_firm_flat::balanced::sponge_v8c_far_fast_early_stop -> sponge_v8c_far_fast_early_stop | 5 | 0.000150 |
| tracer_sponge_firm_flat::balanced::sponge_v8c_mid_fast_early_stop -> sponge_v8c_far_fast_early_stop | 5 | 0.000207 |
| tracer_sponge_firm_flat::balanced::sponge_v8c_soft_stop -> sponge_v8c_far_fast_early_stop | 5 | 0.000152 |
| tracer_sponge_firm_flat::balanced::trot_cautious -> sponge_v8c_far_fast_early_stop | 16 | 0.000574 |
| tracer_sponge_firm_flat::balanced::trot_mid -> sponge_v8c_far_fast_early_stop | 17 | 0.000137 |
| tracer_sponge_firm_flat::balanced::trot_soft_mid_clear -> sponge_v8c_far_fast_early_stop | 32 | 0.000030 |
| tracer_sponge_firm_flat::balanced::trot_solid_fast -> sponge_v8c_far_fast_early_stop | 10 | 0.000025 |
| tracer_sponge_firm_flat::reach::sponge_mid_brake_clear -> sponge_v8b_probe_reach_fast | 5 | 0.000358 |
| tracer_sponge_firm_flat::reach::sponge_probe_crawlish -> sponge_v8b_probe_reach_fast | 19 | 0.000011 |
| tracer_sponge_firm_flat::reach::sponge_reach_then_brake -> sponge_v8b_probe_reach_fast | 22 | 0.000090 |
| tracer_sponge_firm_flat::reach::sponge_short_step_stable -> sponge_v8b_probe_reach_fast | 5 | 0.001238 |
| tracer_sponge_firm_flat::reach::sponge_slow_high_clear -> sponge_v8b_probe_reach_fast | 5 | 0.000210 |
| tracer_sponge_firm_flat::reach::sponge_v8_anchor_mix_top3 -> sponge_v8b_probe_reach_fast | 10 | 0.000319 |
| tracer_sponge_firm_flat::reach::sponge_v8b_probe_reach_fast -> sponge_v8b_probe_reach_fast | 5 | 0.000360 |
| tracer_sponge_firm_flat::reach::sponge_v8b_reach_bias -> sponge_v8b_probe_reach_fast | 5 | 0.000159 |
| tracer_sponge_firm_flat::reach::sponge_v8b_reach_bias_fast -> sponge_v8b_probe_reach_fast | 10 | 0.000100 |
| tracer_sponge_firm_flat::reach::sponge_v8b_reach_stabilized -> sponge_v8b_probe_reach_fast | 5 | 0.000137 |
| tracer_sponge_firm_flat::reach::sponge_v8b_soft_reach -> sponge_v8b_probe_reach_fast | 5 | 0.000346 |
| tracer_sponge_firm_flat::reach::sponge_v8c_far_fast_early_stop -> sponge_v8b_probe_reach_fast | 5 | 0.000070 |
| tracer_sponge_firm_flat::reach::sponge_v8c_mid_fast_early_stop -> sponge_v8b_probe_reach_fast | 5 | 0.000336 |
| tracer_sponge_firm_flat::reach::sponge_v8c_soft_stop -> sponge_v8b_probe_reach_fast | 5 | 0.000837 |
| tracer_sponge_firm_flat::reach::trot_cautious -> sponge_v8b_probe_reach_fast | 16 | 0.000366 |
| tracer_sponge_firm_flat::reach::trot_mid -> sponge_v8b_probe_reach_fast | 17 | 0.000638 |
| tracer_sponge_firm_flat::reach::trot_soft_mid_clear -> sponge_v8b_probe_reach_fast | 32 | 0.000014 |
| tracer_sponge_firm_flat::reach::trot_solid_fast -> sponge_v8b_probe_reach_fast | 10 | 0.000018 |
| tracer_sponge_firm_flat::stability::sponge_mid_brake_clear -> sponge_v8_anchor_mix_top3 | 5 | 0.000245 |
| tracer_sponge_firm_flat::stability::sponge_probe_crawlish -> sponge_v8_anchor_mix_top3 | 19 | 0.000015 |
| tracer_sponge_firm_flat::stability::sponge_reach_then_brake -> sponge_v8_anchor_mix_top3 | 22 | 0.000036 |
| tracer_sponge_firm_flat::stability::sponge_short_step_stable -> sponge_v8_anchor_mix_top3 | 5 | 0.000883 |
| tracer_sponge_firm_flat::stability::sponge_slow_high_clear -> sponge_v8_anchor_mix_top3 | 5 | 0.000239 |
| tracer_sponge_firm_flat::stability::sponge_v8_anchor_mix_top3 -> sponge_v8_anchor_mix_top3 | 10 | 0.000355 |
| tracer_sponge_firm_flat::stability::sponge_v8b_probe_reach_fast -> sponge_v8_anchor_mix_top3 | 5 | 0.000132 |
| tracer_sponge_firm_flat::stability::sponge_v8b_reach_bias -> sponge_v8_anchor_mix_top3 | 5 | 0.000085 |
| tracer_sponge_firm_flat::stability::sponge_v8b_reach_bias_fast -> sponge_v8_anchor_mix_top3 | 10 | 0.000041 |
| tracer_sponge_firm_flat::stability::sponge_v8b_reach_stabilized -> sponge_v8_anchor_mix_top3 | 5 | 0.000047 |
| tracer_sponge_firm_flat::stability::sponge_v8b_soft_reach -> sponge_v8_anchor_mix_top3 | 5 | 0.000407 |
| tracer_sponge_firm_flat::stability::sponge_v8c_far_fast_early_stop -> sponge_v8_anchor_mix_top3 | 5 | 0.000102 |
| tracer_sponge_firm_flat::stability::sponge_v8c_mid_fast_early_stop -> sponge_v8_anchor_mix_top3 | 5 | 0.000126 |
| tracer_sponge_firm_flat::stability::sponge_v8c_soft_stop -> sponge_v8_anchor_mix_top3 | 5 | 0.000451 |
| tracer_sponge_firm_flat::stability::trot_cautious -> sponge_v8_anchor_mix_top3 | 16 | 0.000175 |
| tracer_sponge_firm_flat::stability::trot_mid -> sponge_v8_anchor_mix_top3 | 17 | 0.000150 |
| tracer_sponge_firm_flat::stability::trot_soft_mid_clear -> sponge_v8_anchor_mix_top3 | 32 | 0.000059 |
| tracer_sponge_firm_flat::stability::trot_solid_fast -> sponge_v8_anchor_mix_top3 | 10 | 0.000706 |
