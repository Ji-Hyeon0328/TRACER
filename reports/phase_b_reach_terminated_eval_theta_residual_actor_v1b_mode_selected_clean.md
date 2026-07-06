# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_theta_residual_actor_v1b_mode_selected_clean`
- Result root: `artifacts/phase_b_theta_residual_actor_v1b_selected_clean_eval`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 10 | 1.000 | 0.148 | 0.198 | 0.051 | 9.112 | 2.316 | 7.613 |
| stairs_single | 10 | 0.900 | 0.078 | 0.399 | 0.320 | 9.109 | 1.675 | 6.898 |
| tracer_sponge_firm_flat | 10 | 0.300 | 0.344 | 0.941 | 0.597 | 3.889 | 0.916 | 2.635 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::theta_actor_v1b_earth_stability_from_trot_solid_fast | 10 | 1.000 | 0.148 | 0.198 | 0.051 | 9.112 | 2.316 |
| stairs_single::theta_actor_v1b_stairs_reach_from_trot_mid | 10 | 0.900 | 0.078 | 0.399 | 0.320 | 9.109 | 1.675 |
| tracer_sponge_firm_flat::theta_actor_v1b_sponge_stability_from_trot_soft_mid_clear | 10 | 0.300 | 0.344 | 0.941 | 0.597 | 3.889 | 0.916 |
