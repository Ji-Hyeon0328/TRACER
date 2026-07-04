# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_proxy_ppo_argmax_v1`
- Result root: `artifacts/phase_b_ppo_policy_smoke_online_v1_tracer_proxy`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 3 | 1.000 | 0.148 | 0.187 | 0.039 | 9.110 | 2.514 | 7.643 |
| stairs_single | 3 | 0.667 | 0.150 | 1.008 | 0.858 | 7.605 | 0.756 | 5.599 |
| tracer_sponge_firm_flat | 3 | 1.000 | 0.067 | 3.822 | 3.755 | 9.596 | -3.043 | 4.681 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 3 | 1.000 | 0.148 | 0.187 | 0.039 | 9.110 | 2.514 |
| stairs_single::trot_solid_fast | 3 | 0.667 | 0.150 | 1.008 | 0.858 | 7.605 | 0.756 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 3 | 1.000 | 0.067 | 3.822 | 3.755 | 9.596 | -3.043 |
