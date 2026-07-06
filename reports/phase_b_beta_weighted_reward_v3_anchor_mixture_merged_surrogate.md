# Phase-B β-weighted Reward Report v2 Surrogate

- Rollouts: `data/phase_b_training_pipeline_v6_anchor_mixture_merged/phase_b_rollout_index_v1.jsonl`
- Beta model: `artifacts/phase_b_objective_selector_beta_v2_anchor_mixture_merged/model.pt`
- Rows: `258`
- Hydrated rows: `258`

| world::action | n | reach | beta_v | beta_s | beta_e | Rv_surr | Rs_surr | Re_surr | score_beta | proxy | final | drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 12 | 1.000 | 0.600 | 0.230 | 0.170 | 7.533 | 1.071 | 0.000 | 4.767 | 7.926 | 0.203 | 0.056 |
| earth::trot_solid_fast | 14 | 0.857 | 0.600 | 0.230 | 0.170 | 6.507 | 0.714 | -0.250 | 4.027 | 6.813 | 0.320 | 0.069 |
| earth::trot_cautious | 5 | 0.200 | 0.600 | 0.230 | 0.170 | 3.283 | 1.253 | 0.000 | 2.258 | 3.711 | 0.172 | 0.000 |
| earth::trot_soft_mid_clear | 5 | 0.000 | 0.600 | 0.230 | 0.170 | 1.473 | 1.073 | -0.150 | 1.105 | 1.871 | 0.253 | 0.000 |
| stairs_single::trot_solid_fast | 16 | 0.938 | 0.461 | 0.399 | 0.141 | 7.858 | -0.363 | -0.250 | 3.440 | 7.966 | 0.486 | 0.382 |
| stairs_single::trot_mid | 10 | 0.600 | 0.461 | 0.399 | 0.141 | 5.589 | -0.257 | 0.000 | 2.472 | 5.692 | 0.503 | 0.265 |
| stairs_single::trot_cautious | 5 | 0.400 | 0.461 | 0.399 | 0.141 | 3.906 | 0.519 | 0.000 | 2.006 | 4.163 | 0.365 | 0.067 |
| stairs_single::trot_soft_mid_clear | 5 | 0.200 | 0.461 | 0.399 | 0.141 | 2.303 | 0.511 | -0.150 | 1.244 | 2.561 | 0.403 | 0.037 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 5 | 0.200 | 0.140 | 0.796 | 0.063 | 2.630 | -9.010 | -0.150 | -6.813 | 0.446 | 2.768 | 0.716 |
| tracer_sponge_firm_flat::sponge_v8_anchor_mix_top3 | 10 | 0.400 | 0.140 | 0.796 | 0.063 | 3.820 | -10.768 | 0.000 | -8.036 | 1.449 | 2.940 | 1.462 |
| tracer_sponge_firm_flat::sponge_v8c_far_fast_early_stop | 5 | 0.600 | 0.140 | 0.796 | 0.063 | 4.869 | -11.046 | 0.000 | -8.110 | 2.688 | 2.788 | 2.090 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 19 | 0.474 | 0.140 | 0.796 | 0.063 | 4.619 | -11.534 | 0.000 | -8.534 | 2.102 | 3.100 | 1.656 |
| tracer_sponge_firm_flat::sponge_v8b_soft_reach | 5 | 0.400 | 0.140 | 0.796 | 0.063 | 3.903 | -11.731 | 0.000 | -8.792 | 1.266 | 3.193 | 1.491 |
| tracer_sponge_firm_flat::sponge_short_step_stable | 5 | 0.400 | 0.140 | 0.796 | 0.063 | 4.089 | -11.966 | 0.000 | -8.952 | 1.367 | 3.282 | 1.458 |
| tracer_sponge_firm_flat::trot_cautious | 16 | 0.562 | 0.140 | 0.796 | 0.063 | 4.917 | -12.193 | 0.000 | -9.017 | 2.364 | 3.145 | 2.003 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 5 | 0.600 | 0.140 | 0.796 | 0.063 | 5.543 | -12.416 | 0.000 | -9.107 | 2.945 | 3.140 | 2.074 |
| tracer_sponge_firm_flat::sponge_mid_brake_clear | 5 | 0.400 | 0.140 | 0.796 | 0.063 | 4.372 | -12.270 | 0.000 | -9.155 | 1.539 | 3.397 | 1.417 |
| tracer_sponge_firm_flat::trot_solid_fast | 10 | 0.400 | 0.140 | 0.796 | 0.063 | 3.970 | -12.571 | -0.250 | -9.467 | 1.116 | 3.426 | 1.546 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 32 | 0.656 | 0.140 | 0.796 | 0.063 | 5.739 | -13.322 | -0.150 | -9.810 | 2.958 | 3.340 | 2.244 |
| tracer_sponge_firm_flat::sponge_v8c_soft_stop | 5 | 0.600 | 0.140 | 0.796 | 0.063 | 5.182 | -13.301 | 0.000 | -9.862 | 2.400 | 3.323 | 2.215 |
| tracer_sponge_firm_flat::trot_mid | 17 | 0.529 | 0.140 | 0.796 | 0.063 | 4.971 | -13.322 | 0.000 | -9.908 | 2.128 | 3.406 | 2.092 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias_fast | 10 | 0.500 | 0.140 | 0.796 | 0.063 | 4.757 | -13.329 | 0.000 | -9.944 | 1.860 | 3.455 | 1.968 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 22 | 0.682 | 0.140 | 0.796 | 0.063 | 6.077 | -14.741 | 0.000 | -10.882 | 2.986 | 3.643 | 2.480 |
| tracer_sponge_firm_flat::sponge_v8b_reach_stabilized | 5 | 0.800 | 0.140 | 0.796 | 0.063 | 6.874 | -16.267 | 0.000 | -11.986 | 3.612 | 3.783 | 3.114 |
| tracer_sponge_firm_flat::sponge_v8b_probe_reach_fast | 5 | 0.800 | 0.140 | 0.796 | 0.063 | 6.988 | -16.369 | 0.000 | -12.051 | 3.690 | 3.851 | 3.101 |
| tracer_sponge_firm_flat::sponge_v8c_mid_fast_early_stop | 5 | 0.800 | 0.140 | 0.796 | 0.063 | 6.198 | -16.317 | 0.000 | -12.120 | 2.921 | 3.821 | 3.072 |
