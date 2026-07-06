# Phase-B β-aware Candidate Policy v7 Deployable

- Rows: `258`
- Input dim: `18`
- Best epoch: `316`
- Best val MSE: `0.002828`

## Group summary

| world::action | n | pred | target_task_beta | proxy | reach | final | drift | beta_v | beta_s | beta_e |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 12 | 1.002 | 1.000 | 7.926 | 1.000 | 0.203 | 0.056 | 0.600 | 0.230 | 0.170 |
| earth::trot_solid_fast | 14 | 0.813 | 0.816 | 6.813 | 0.857 | 0.320 | 0.069 | 0.600 | 0.230 | 0.170 |
| earth::trot_cautious | 5 | 0.291 | 0.291 | 3.711 | 0.200 | 0.172 | 0.000 | 0.600 | 0.230 | 0.170 |
| earth::trot_soft_mid_clear | 5 | -0.005 | 0.000 | 1.871 | 0.000 | 0.253 | 0.000 | 0.600 | 0.230 | 0.170 |
| stairs_single::trot_solid_fast | 16 | 1.001 | 1.000 | 7.966 | 0.938 | 0.486 | 0.382 | 0.461 | 0.399 | 0.141 |
| stairs_single::trot_mid | 10 | 0.566 | 0.566 | 5.692 | 0.600 | 0.503 | 0.265 | 0.461 | 0.399 | 0.141 |
| stairs_single::trot_cautious | 5 | 0.309 | 0.310 | 4.163 | 0.400 | 0.365 | 0.067 | 0.461 | 0.399 | 0.141 |
| stairs_single::trot_soft_mid_clear | 5 | -0.001 | 0.000 | 2.561 | 0.200 | 0.403 | 0.037 | 0.461 | 0.399 | 0.141 |
| tracer_sponge_firm_flat::sponge_v8c_far_fast_early_stop | 5 | 0.740 | 0.710 | 2.688 | 0.600 | 2.788 | 2.090 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 5 | 0.695 | 0.683 | 2.945 | 0.600 | 3.140 | 2.074 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 32 | 0.670 | 0.653 | 2.958 | 0.656 | 3.340 | 2.244 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::sponge_v8b_reach_stabilized | 5 | 0.663 | 0.647 | 3.612 | 0.800 | 3.783 | 3.114 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::sponge_v8b_probe_reach_fast | 5 | 0.647 | 0.655 | 3.690 | 0.800 | 3.851 | 3.101 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 22 | 0.603 | 0.594 | 2.986 | 0.682 | 3.643 | 2.480 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::trot_cautious | 16 | 0.594 | 0.591 | 2.364 | 0.562 | 3.145 | 2.003 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 19 | 0.554 | 0.560 | 2.102 | 0.474 | 3.100 | 1.656 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::sponge_v8c_mid_fast_early_stop | 5 | 0.545 | 0.535 | 2.921 | 0.800 | 3.821 | 3.072 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::sponge_v8c_soft_stop | 5 | 0.524 | 0.551 | 2.400 | 0.600 | 3.323 | 2.215 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::trot_mid | 17 | 0.477 | 0.488 | 2.128 | 0.529 | 3.406 | 2.092 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::sponge_v8_anchor_mix_top3 | 10 | 0.462 | 0.474 | 1.449 | 0.400 | 2.940 | 1.462 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias_fast | 10 | 0.446 | 0.437 | 1.860 | 0.500 | 3.455 | 1.968 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::sponge_mid_brake_clear | 5 | 0.406 | 0.414 | 1.539 | 0.400 | 3.397 | 1.417 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::sponge_short_step_stable | 5 | 0.376 | 0.402 | 1.367 | 0.400 | 3.282 | 1.458 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::sponge_v8b_soft_reach | 5 | 0.372 | 0.397 | 1.266 | 0.400 | 3.193 | 1.491 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 5 | 0.315 | 0.350 | 0.446 | 0.200 | 2.768 | 0.716 | 0.140 | 0.796 | 0.063 |
| tracer_sponge_firm_flat::trot_solid_fast | 10 | 0.312 | 0.330 | 1.116 | 0.400 | 3.426 | 1.546 | 0.140 | 0.796 | 0.063 |
