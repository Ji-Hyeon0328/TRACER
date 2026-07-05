# Phase-B β-weighted Reward Report v2 Surrogate

- Rollouts: `data/phase_b_training_pipeline_v5_merged_balanced/phase_b_rollout_index_v1.jsonl`
- Beta model: `artifacts/phase_b_objective_selector_beta_v1_hydrated/model.pt`
- Rows: `173`
- Hydrated rows: `173`

| world::action | n | reach | beta_v | beta_s | beta_e | Rv_surr | Rs_surr | Re_surr | score_beta | proxy | final | drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 12 | 1.000 | 0.600 | 0.231 | 0.169 | 7.533 | 1.071 | 0.000 | 4.767 | 7.926 | 0.203 | 0.056 |
| earth::trot_solid_fast | 14 | 0.857 | 0.600 | 0.231 | 0.169 | 6.507 | 0.714 | -0.250 | 4.027 | 6.813 | 0.320 | 0.069 |
| earth::trot_cautious | 5 | 0.200 | 0.600 | 0.231 | 0.169 | 3.283 | 1.253 | 0.000 | 2.259 | 3.711 | 0.172 | 0.000 |
| earth::trot_soft_mid_clear | 5 | 0.000 | 0.600 | 0.231 | 0.169 | 1.473 | 1.073 | -0.150 | 1.106 | 1.871 | 0.253 | 0.000 |
| stairs_single::trot_solid_fast | 16 | 0.938 | 0.465 | 0.399 | 0.136 | 7.858 | -0.363 | -0.250 | 3.473 | 7.966 | 0.486 | 0.382 |
| stairs_single::trot_mid | 10 | 0.600 | 0.465 | 0.399 | 0.136 | 5.589 | -0.257 | 0.000 | 2.495 | 5.692 | 0.503 | 0.265 |
| stairs_single::trot_cautious | 5 | 0.400 | 0.465 | 0.399 | 0.136 | 3.906 | 0.519 | 0.000 | 2.023 | 4.163 | 0.365 | 0.067 |
| stairs_single::trot_soft_mid_clear | 5 | 0.200 | 0.465 | 0.399 | 0.136 | 2.303 | 0.511 | -0.150 | 1.254 | 2.561 | 0.403 | 0.037 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 5 | 0.200 | 0.142 | 0.795 | 0.063 | 2.630 | -9.010 | -0.150 | -6.799 | 0.446 | 2.768 | 0.716 |
| tracer_sponge_firm_flat::sponge_short_step_stable | 5 | 0.400 | 0.142 | 0.795 | 0.063 | 4.089 | -11.966 | 0.000 | -8.933 | 1.367 | 3.282 | 1.458 |
| tracer_sponge_firm_flat::trot_cautious | 16 | 0.562 | 0.142 | 0.795 | 0.063 | 4.917 | -12.193 | 0.000 | -8.996 | 2.364 | 3.145 | 2.003 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 14 | 0.571 | 0.142 | 0.795 | 0.063 | 5.334 | -12.287 | 0.000 | -9.012 | 2.740 | 3.180 | 1.991 |
| tracer_sponge_firm_flat::sponge_mid_brake_clear | 5 | 0.400 | 0.142 | 0.795 | 0.063 | 4.372 | -12.270 | 0.000 | -9.134 | 1.539 | 3.397 | 1.417 |
| tracer_sponge_firm_flat::trot_solid_fast | 10 | 0.400 | 0.142 | 0.795 | 0.063 | 3.970 | -12.571 | -0.250 | -9.446 | 1.116 | 3.426 | 1.546 |
| tracer_sponge_firm_flat::trot_mid | 17 | 0.529 | 0.142 | 0.795 | 0.063 | 4.971 | -13.322 | 0.000 | -9.885 | 2.128 | 3.406 | 2.092 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 12 | 0.750 | 0.142 | 0.795 | 0.063 | 6.361 | -14.817 | 0.000 | -10.877 | 3.316 | 3.598 | 2.637 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 17 | 0.765 | 0.142 | 0.795 | 0.063 | 6.542 | -15.264 | -0.150 | -11.216 | 3.426 | 3.656 | 2.775 |
