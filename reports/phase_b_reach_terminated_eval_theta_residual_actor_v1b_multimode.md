# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_theta_residual_actor_v1b_multimode`
- Result root: `artifacts/phase_b_theta_residual_actor_v1b_multimode_eval`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 15 | 0.933 | 0.170 | 0.249 | 0.079 | 8.581 | 2.336 | 7.187 |
| stairs_single | 15 | 0.933 | 0.091 | 0.511 | 0.420 | 9.177 | 1.592 | 6.934 |
| tracer_sponge_firm_flat | 15 | 0.533 | 0.167 | 3.340 | 3.173 | 6.723 | -2.678 | 3.078 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::theta_actor_v1b_earth_balanced_from_trot_solid_fast | 5 | 0.800 | 0.217 | 0.296 | 0.079 | 7.502 | 2.132 |
| earth::theta_actor_v1b_earth_reach_from_trot_mid | 5 | 1.000 | 0.146 | 0.247 | 0.102 | 9.126 | 2.309 |
| earth::theta_actor_v1b_earth_stability_from_trot_solid_fast | 5 | 1.000 | 0.147 | 0.203 | 0.056 | 9.116 | 2.568 |
| stairs_single::theta_actor_v1b_stairs_balanced_from_trot_mid | 5 | 1.000 | 0.113 | 0.554 | 0.440 | 9.319 | 1.453 |
| stairs_single::theta_actor_v1b_stairs_reach_from_trot_mid | 5 | 1.000 | 0.067 | 0.601 | 0.534 | 9.595 | 1.575 |
| stairs_single::theta_actor_v1b_stairs_stability_from_trot_solid_fast | 5 | 0.800 | 0.093 | 0.378 | 0.285 | 8.616 | 1.746 |
| tracer_sponge_firm_flat::theta_actor_v1b_sponge_balanced_from_trot_soft_mid_clear | 5 | 0.400 | 0.177 | 3.322 | 3.145 | 6.135 | -2.793 |
| tracer_sponge_firm_flat::theta_actor_v1b_sponge_reach_from_trot_soft_mid_clear | 5 | 0.400 | 0.192 | 3.358 | 3.166 | 6.004 | -3.019 |
| tracer_sponge_firm_flat::theta_actor_v1b_sponge_stability_from_trot_soft_mid_clear | 5 | 0.800 | 0.133 | 3.341 | 3.208 | 8.030 | -2.222 |
