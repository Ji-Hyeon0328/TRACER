# Phase-B Reach-Terminated Evaluation

- Policy: `phase_b_ablation_data_v1_stairs`
- Result root: `artifacts/phase_b_ablation_data_v1/stairs`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stairs_single | 60 | 0.967 | 0.097 | 0.249 | 0.152 | 9.250 | 1.900 | 7.059 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| stairs_single::trot_mid | 30 | 0.933 | 0.113 | 0.251 | 0.138 | 8.986 | 1.857 |
| stairs_single::trot_solid_fast | 30 | 1.000 | 0.080 | 0.246 | 0.166 | 9.515 | 1.943 |
