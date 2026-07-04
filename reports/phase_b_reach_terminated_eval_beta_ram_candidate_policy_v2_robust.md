# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_beta_ram_candidate_policy_v2_robust`
- Result root: `artifacts/phase_b_beta_ram_candidate_policy_v2_robust_eval`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 3 | 1.000 | 0.136 | 0.595 | 0.459 | 9.181 | 1.642 | 7.564 |
| stairs_single | 3 | 1.000 | 0.060 | 0.627 | 0.566 | 9.638 | 1.644 | 7.276 |
| tracer_sponge_firm_flat | 3 | 0.000 | 0.194 | 3.607 | 3.413 | 4.571 | -3.405 | 1.516 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 3 | 1.000 | 0.136 | 0.595 | 0.459 | 9.181 | 1.642 |
| stairs_single::trot_solid_fast | 3 | 1.000 | 0.060 | 0.627 | 0.566 | 9.638 | 1.644 |
| tracer_sponge_firm_flat::trot_mid | 3 | 0.000 | 0.194 | 3.607 | 3.413 | 4.571 | -3.405 |
