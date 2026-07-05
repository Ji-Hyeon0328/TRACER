# Phase-B Task-aware β Reward Report v0

- Beta report: `reports/phase_b_beta_weighted_reward_v2_surrogate.json`
- Task weight: `0.65`
- Beta weight: `0.35`

`score_task_beta = task_weight * task_norm + beta_weight * beta_score_norm`

| world::action | n | reach | proxy | beta_score | task_norm | beta_norm | score_task_beta | final | drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 12 | 1.000 | 7.926 | 4.767 | 1.000 | 1.000 | 1.000 | 0.203 | 0.056 |
| earth::trot_solid_fast | 14 | 0.857 | 6.813 | 4.027 | 0.826 | 0.798 | 0.816 | 0.320 | 0.069 |
| earth::trot_cautious | 5 | 0.200 | 3.711 | 2.259 | 0.278 | 0.315 | 0.291 | 0.172 | 0.000 |
| earth::trot_soft_mid_clear | 5 | 0.000 | 1.871 | 1.106 | 0.000 | 0.000 | 0.000 | 0.253 | 0.000 |
| stairs_single::trot_solid_fast | 16 | 0.938 | 7.966 | 3.473 | 1.000 | 1.000 | 1.000 | 0.486 | 0.382 |
| stairs_single::trot_mid | 10 | 0.600 | 5.692 | 2.495 | 0.570 | 0.559 | 0.566 | 0.503 | 0.265 |
| stairs_single::trot_cautious | 5 | 0.400 | 4.163 | 2.023 | 0.290 | 0.346 | 0.310 | 0.365 | 0.067 |
| stairs_single::trot_soft_mid_clear | 5 | 0.200 | 2.561 | 1.254 | 0.000 | 0.000 | 0.000 | 0.403 | 0.037 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 14 | 0.571 | 2.740 | -9.012 | 0.742 | 0.499 | 0.657 | 3.180 | 1.991 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 12 | 0.750 | 3.316 | -10.877 | 0.966 | 0.077 | 0.655 | 3.598 | 2.637 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 17 | 0.765 | 3.426 | -11.216 | 1.000 | 0.000 | 0.650 | 3.656 | 2.775 |
| tracer_sponge_firm_flat::trot_cautious | 16 | 0.562 | 2.364 | -8.996 | 0.643 | 0.503 | 0.594 | 3.145 | 2.003 |
| tracer_sponge_firm_flat::trot_mid | 17 | 0.529 | 2.128 | -9.885 | 0.569 | 0.301 | 0.475 | 3.406 | 2.092 |
| tracer_sponge_firm_flat::sponge_mid_brake_clear | 5 | 0.400 | 1.539 | -9.134 | 0.364 | 0.471 | 0.401 | 3.397 | 1.417 |
| tracer_sponge_firm_flat::sponge_short_step_stable | 5 | 0.400 | 1.367 | -8.933 | 0.320 | 0.517 | 0.389 | 3.282 | 1.458 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 5 | 0.200 | 0.446 | -6.799 | 0.000 | 1.000 | 0.350 | 2.768 | 0.716 |
| tracer_sponge_firm_flat::trot_solid_fast | 10 | 0.400 | 1.116 | -9.446 | 0.257 | 0.401 | 0.307 | 3.426 | 1.546 |
