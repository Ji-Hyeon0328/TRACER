# Phase-B Task-aware β Reward Report v0

- Beta report: `reports/phase_b_beta_weighted_reward_v3_anchor_mixture_merged_surrogate.json`
- Task weight: `0.65`
- Beta weight: `0.35`

`score_task_beta = task_weight * task_norm + beta_weight * beta_score_norm`

| world::action | n | reach | proxy | beta_score | task_norm | beta_norm | score_task_beta | final | drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 12 | 1.000 | 7.926 | 4.767 | 1.000 | 1.000 | 1.000 | 0.203 | 0.056 |
| earth::trot_solid_fast | 14 | 0.857 | 6.813 | 4.027 | 0.826 | 0.798 | 0.816 | 0.320 | 0.069 |
| earth::trot_cautious | 5 | 0.200 | 3.711 | 2.258 | 0.278 | 0.315 | 0.291 | 0.172 | 0.000 |
| earth::trot_soft_mid_clear | 5 | 0.000 | 1.871 | 1.105 | 0.000 | 0.000 | 0.000 | 0.253 | 0.000 |
| stairs_single::trot_solid_fast | 16 | 0.938 | 7.966 | 3.440 | 1.000 | 1.000 | 1.000 | 0.486 | 0.382 |
| stairs_single::trot_mid | 10 | 0.600 | 5.692 | 2.472 | 0.570 | 0.559 | 0.566 | 0.503 | 0.265 |
| stairs_single::trot_cautious | 5 | 0.400 | 4.163 | 2.006 | 0.290 | 0.347 | 0.310 | 0.365 | 0.067 |
| stairs_single::trot_soft_mid_clear | 5 | 0.200 | 2.561 | 1.244 | 0.000 | 0.000 | 0.000 | 0.403 | 0.037 |
| tracer_sponge_firm_flat::sponge_v8c_far_fast_early_stop | 5 | 0.600 | 2.688 | -8.110 | 0.685 | 0.756 | 0.710 | 2.788 | 2.090 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 5 | 0.600 | 2.945 | -9.107 | 0.744 | 0.568 | 0.683 | 3.140 | 2.074 |
| tracer_sponge_firm_flat::sponge_v8b_probe_reach_fast | 5 | 0.800 | 3.690 | -12.051 | 1.000 | 0.013 | 0.655 | 3.851 | 3.101 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 32 | 0.656 | 2.958 | -9.810 | 0.771 | 0.435 | 0.653 | 3.340 | 2.244 |
| tracer_sponge_firm_flat::sponge_v8b_reach_stabilized | 5 | 0.800 | 3.612 | -11.986 | 0.982 | 0.025 | 0.647 | 3.783 | 3.114 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 22 | 0.682 | 2.986 | -10.882 | 0.788 | 0.233 | 0.594 | 3.643 | 2.480 |
| tracer_sponge_firm_flat::trot_cautious | 16 | 0.562 | 2.364 | -9.017 | 0.595 | 0.585 | 0.591 | 3.145 | 2.003 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 19 | 0.474 | 2.102 | -8.534 | 0.497 | 0.676 | 0.560 | 3.100 | 1.656 |
| tracer_sponge_firm_flat::sponge_v8c_soft_stop | 5 | 0.600 | 2.400 | -9.862 | 0.619 | 0.426 | 0.551 | 3.323 | 2.215 |
| tracer_sponge_firm_flat::sponge_v8c_mid_fast_early_stop | 5 | 0.800 | 2.921 | -12.120 | 0.822 | 0.000 | 0.535 | 3.821 | 3.072 |
| tracer_sponge_firm_flat::trot_mid | 17 | 0.529 | 2.128 | -9.908 | 0.526 | 0.417 | 0.488 | 3.406 | 2.092 |
| tracer_sponge_firm_flat::sponge_v8_anchor_mix_top3 | 10 | 0.400 | 1.449 | -8.036 | 0.315 | 0.770 | 0.474 | 2.940 | 1.462 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias_fast | 10 | 0.500 | 1.860 | -9.944 | 0.452 | 0.410 | 0.437 | 3.455 | 1.968 |
| tracer_sponge_firm_flat::sponge_mid_brake_clear | 5 | 0.400 | 1.539 | -9.155 | 0.336 | 0.559 | 0.414 | 3.397 | 1.417 |
| tracer_sponge_firm_flat::sponge_short_step_stable | 5 | 0.400 | 1.367 | -8.952 | 0.296 | 0.597 | 0.402 | 3.282 | 1.458 |
| tracer_sponge_firm_flat::sponge_v8b_soft_reach | 5 | 0.400 | 1.266 | -8.792 | 0.273 | 0.627 | 0.397 | 3.193 | 1.491 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 5 | 0.200 | 0.446 | -6.813 | 0.000 | 1.000 | 0.350 | 2.768 | 0.716 |
| tracer_sponge_firm_flat::trot_solid_fast | 10 | 0.400 | 1.116 | -9.467 | 0.238 | 0.500 | 0.330 | 3.426 | 1.546 |
