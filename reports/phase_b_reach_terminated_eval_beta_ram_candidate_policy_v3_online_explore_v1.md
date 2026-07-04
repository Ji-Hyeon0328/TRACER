# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_beta_ram_candidate_policy_v3_online_explore_v1`
- Result root: `artifacts/phase_b_beta_ram_candidate_policy_v3_online_explore_v1`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 6 | 0.667 | 0.220 | 0.437 | 0.218 | 6.938 | 1.909 | 5.799 |
| stairs_single | 6 | 1.000 | 0.089 | 0.431 | 0.342 | 9.465 | 1.685 | 7.157 |
| tracer_sponge_firm_flat | 6 | 0.667 | 0.140 | 2.678 | 2.537 | 7.561 | -1.928 | 3.842 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 2 | 1.000 | 0.147 | 0.212 | 0.065 | 9.118 | 2.363 |
| earth::trot_solid_fast | 4 | 0.500 | 0.256 | 0.550 | 0.294 | 5.848 | 1.682 |
| stairs_single::trot_solid_fast | 6 | 1.000 | 0.089 | 0.431 | 0.342 | 9.465 | 1.685 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 3 | 0.333 | 0.200 | 2.384 | 2.184 | 5.606 | -1.521 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 1 | 1.000 | 0.132 | 3.928 | 3.796 | 9.209 | -3.619 |
| tracer_sponge_firm_flat::trot_mid | 1 | 1.000 | 0.108 | 3.769 | 3.661 | 9.352 | -3.512 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 1 | 1.000 | 0.002 | 1.216 | 1.214 | 9.985 | 0.126 |
