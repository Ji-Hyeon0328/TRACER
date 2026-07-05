# Phase-B β-weighted Reward Report v1 Hydrated

- Rollouts: `data/phase_b_training_pipeline_v5_merged_balanced/phase_b_rollout_index_v1.jsonl`
- Beta model: `artifacts/phase_b_objective_selector_beta_v1_hydrated/model.pt`
- Rows: `173`
- Hydrated rows: `173`

| world::action | n | reach | beta_v | beta_s | beta_e | R_v | R_s | R_e | score_beta | proxy_reward | final | drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_cautious | 5 | 0.200 | 0.600 | 0.231 | 0.169 | 0.000 | 0.000 | 0.000 | 0.000 | 3.711 | 0.172 | 0.000 |
| earth::trot_mid | 12 | 1.000 | 0.600 | 0.231 | 0.169 | 0.000 | 0.000 | 0.000 | 0.000 | 7.926 | 0.203 | 0.000 |
| earth::trot_soft_mid_clear | 5 | 0.000 | 0.600 | 0.231 | 0.169 | 0.000 | 0.000 | 0.000 | 0.000 | 1.871 | 0.253 | 0.000 |
| earth::trot_solid_fast | 14 | 0.857 | 0.600 | 0.231 | 0.169 | 0.000 | 0.000 | 0.000 | 0.000 | 6.813 | 0.320 | 0.000 |
| stairs_single::trot_cautious | 5 | 0.400 | 0.465 | 0.399 | 0.136 | 0.000 | 0.000 | 0.000 | 0.000 | 4.163 | 0.365 | 0.000 |
| stairs_single::trot_mid | 10 | 0.600 | 0.465 | 0.399 | 0.136 | 0.000 | 0.000 | 0.000 | 0.000 | 5.692 | 0.503 | 0.000 |
| stairs_single::trot_soft_mid_clear | 5 | 0.200 | 0.465 | 0.399 | 0.136 | 0.000 | 0.000 | 0.000 | 0.000 | 2.561 | 0.403 | 0.000 |
| stairs_single::trot_solid_fast | 16 | 0.938 | 0.465 | 0.399 | 0.136 | 0.000 | 0.000 | 0.000 | 0.000 | 7.966 | 0.486 | 0.000 |
| tracer_sponge_firm_flat::sponge_mid_brake_clear | 5 | 0.400 | 0.142 | 0.795 | 0.063 | 0.000 | 0.000 | 0.000 | 0.000 | 1.539 | 3.397 | 0.000 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 14 | 0.571 | 0.142 | 0.795 | 0.063 | 0.000 | 0.000 | 0.000 | 0.000 | 2.740 | 3.180 | 0.000 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 12 | 0.750 | 0.142 | 0.795 | 0.063 | 0.000 | 0.000 | 0.000 | 0.000 | 3.316 | 3.598 | 0.000 |
| tracer_sponge_firm_flat::sponge_short_step_stable | 5 | 0.400 | 0.142 | 0.795 | 0.063 | 0.000 | 0.000 | 0.000 | 0.000 | 1.367 | 3.282 | 0.000 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 5 | 0.200 | 0.142 | 0.795 | 0.063 | 0.000 | 0.000 | 0.000 | 0.000 | 0.446 | 2.768 | 0.000 |
| tracer_sponge_firm_flat::trot_cautious | 16 | 0.562 | 0.142 | 0.795 | 0.063 | 0.000 | 0.000 | 0.000 | 0.000 | 2.364 | 3.145 | 0.000 |
| tracer_sponge_firm_flat::trot_mid | 17 | 0.529 | 0.142 | 0.795 | 0.063 | 0.000 | 0.000 | 0.000 | 0.000 | 2.128 | 3.406 | 0.000 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 17 | 0.765 | 0.142 | 0.795 | 0.063 | 0.000 | 0.000 | 0.000 | 0.000 | 3.426 | 3.656 | 0.000 |
| tracer_sponge_firm_flat::trot_solid_fast | 10 | 0.400 | 0.142 | 0.795 | 0.063 | 0.000 | 0.000 | 0.000 | 0.000 | 1.116 | 3.426 | 0.000 |
