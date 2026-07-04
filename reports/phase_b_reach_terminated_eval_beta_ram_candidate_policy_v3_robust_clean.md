# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_beta_ram_candidate_policy_v3_robust_clean`
- Result root: `artifacts/phase_b_beta_ram_candidate_policy_v3_robust_clean_eval`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 3 | 1.000 | 0.148 | 0.219 | 0.071 | 9.111 | 2.442 | 7.633 |
| stairs_single | 3 | 1.000 | 0.045 | 0.353 | 0.308 | 9.727 | 2.165 | 7.464 |
| tracer_sponge_firm_flat | 3 | 0.333 | 0.335 | 0.427 | 0.092 | 4.025 | 1.650 | 2.967 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 3 | 1.000 | 0.148 | 0.219 | 0.071 | 9.111 | 2.442 |
| stairs_single::trot_solid_fast | 3 | 1.000 | 0.045 | 0.353 | 0.308 | 9.727 | 2.165 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 3 | 0.333 | 0.335 | 0.427 | 0.092 | 4.025 | 1.650 |
