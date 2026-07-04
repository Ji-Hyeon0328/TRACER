# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_beta_ram_candidate_policy_v5_robust_clean`
- Result root: `artifacts/phase_b_beta_ram_candidate_policy_v5_robust_clean_eval`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 3 | 0.333 | 0.343 | 0.563 | 0.220 | 4.168 | 1.633 | 3.549 |
| stairs_single | 3 | 0.667 | 0.186 | 0.333 | 0.146 | 7.173 | 1.968 | 5.574 |
| tracer_sponge_firm_flat | 3 | 0.333 | 0.212 | 2.385 | 2.173 | 5.547 | -1.565 | 2.759 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 3 | 0.333 | 0.343 | 0.563 | 0.220 | 4.168 | 1.633 |
| stairs_single::trot_solid_fast | 3 | 0.667 | 0.186 | 0.333 | 0.146 | 7.173 | 1.968 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 3 | 0.333 | 0.212 | 2.385 | 2.173 | 5.547 | -1.565 |
