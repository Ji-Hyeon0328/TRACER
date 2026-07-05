# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_beta_aware_candidate_policy_v7_selected`
- Result root: `artifacts/phase_b_beta_aware_candidate_policy_v7_selected_eval`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 5 | 1.000 | 0.147 | 0.199 | 0.052 | 9.115 | 2.379 | 7.626 |
| stairs_single | 5 | 1.000 | 0.070 | 0.280 | 0.210 | 9.577 | 2.033 | 7.320 |
| tracer_sponge_firm_flat | 5 | 0.200 | 0.267 | 2.232 | 1.964 | 4.397 | -1.108 | 2.236 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 5 | 1.000 | 0.147 | 0.199 | 0.052 | 9.115 | 2.379 |
| stairs_single::trot_solid_fast | 5 | 1.000 | 0.070 | 0.280 | 0.210 | 9.577 | 2.033 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 5 | 0.200 | 0.267 | 2.232 | 1.964 | 4.397 | -1.108 |
