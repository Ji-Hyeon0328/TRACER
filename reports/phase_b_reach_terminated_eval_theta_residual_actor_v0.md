# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_theta_residual_actor_v0`
- Result root: `artifacts/phase_b_theta_residual_actor_v0_eval`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 5 | 1.000 | 0.148 | 0.188 | 0.040 | 9.112 | 2.462 | 7.637 |
| stairs_single | 5 | 1.000 | 0.057 | 0.604 | 0.547 | 9.657 | 1.505 | 7.256 |
| tracer_sponge_firm_flat | 15 | 0.600 | 0.196 | 3.270 | 3.074 | 6.746 | -2.514 | 3.154 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::theta_actor_v0_earth_from_trot_solid_fast | 5 | 1.000 | 0.148 | 0.188 | 0.040 | 9.112 | 2.462 |
| stairs_single::theta_actor_v0_stairs_from_trot_mid | 5 | 1.000 | 0.057 | 0.604 | 0.547 | 9.657 | 1.505 |
| tracer_sponge_firm_flat::theta_actor_v0_sponge_from_sponge_probe_crawlish | 5 | 0.800 | 0.161 | 3.740 | 3.580 | 7.915 | -3.244 |
| tracer_sponge_firm_flat::theta_actor_v0_sponge_from_sponge_reach_then_brake | 5 | 0.800 | 0.116 | 3.922 | 3.806 | 8.305 | -3.156 |
| tracer_sponge_firm_flat::theta_actor_v0_sponge_from_trot_soft_mid_clear | 5 | 0.200 | 0.310 | 2.147 | 1.837 | 4.017 | -1.141 |
