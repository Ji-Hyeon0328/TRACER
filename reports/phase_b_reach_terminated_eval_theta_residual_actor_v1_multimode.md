# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_theta_residual_actor_v1_multimode`
- Result root: `artifacts/phase_b_theta_residual_actor_v1_multimode_eval`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 15 | 1.000 | 0.149 | 0.214 | 0.065 | 9.107 | 2.366 | 7.612 |
| stairs_single | 15 | 1.000 | 0.078 | 0.278 | 0.200 | 9.531 | 1.944 | 7.271 |
| tracer_sponge_firm_flat | 15 | 0.467 | 0.238 | 3.133 | 2.895 | 5.811 | -2.466 | 2.603 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::theta_actor_v1_earth_balanced_from_trot_solid_fast | 5 | 1.000 | 0.149 | 0.214 | 0.064 | 9.104 | 2.443 |
| earth::theta_actor_v1_earth_reach_from_trot_mid | 5 | 1.000 | 0.147 | 0.229 | 0.081 | 9.114 | 2.417 |
| earth::theta_actor_v1_earth_stability_from_trot_solid_fast | 5 | 1.000 | 0.149 | 0.200 | 0.051 | 9.102 | 2.239 |
| stairs_single::theta_actor_v1_stairs_balanced_from_trot_mid | 5 | 1.000 | 0.084 | 0.367 | 0.283 | 9.496 | 1.991 |
| stairs_single::theta_actor_v1_stairs_reach_from_trot_mid | 5 | 1.000 | 0.070 | 0.228 | 0.158 | 9.578 | 1.946 |
| stairs_single::theta_actor_v1_stairs_stability_from_trot_solid_fast | 5 | 1.000 | 0.080 | 0.238 | 0.158 | 9.519 | 1.895 |
| tracer_sponge_firm_flat::theta_actor_v1_sponge_balanced_from_trot_soft_mid_clear | 5 | 0.600 | 0.273 | 3.166 | 2.893 | 5.898 | -2.295 |
| tracer_sponge_firm_flat::theta_actor_v1_sponge_reach_from_trot_soft_mid_clear | 5 | 0.200 | 0.273 | 3.116 | 2.844 | 4.425 | -2.741 |
| tracer_sponge_firm_flat::theta_actor_v1_sponge_stability_from_trot_soft_mid_clear | 5 | 0.600 | 0.169 | 3.116 | 2.947 | 7.110 | -2.361 |
