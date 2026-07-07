# Phase-B Reach-Terminated Evaluation

- Policy: `phase_b_ablation_data_v1_earth`
- Result root: `artifacts/phase_b_ablation_data_v1/earth`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 60 | 0.583 | 0.166 | 0.186 | 0.019 | 7.234 | 2.389 | 6.114 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 30 | 0.167 | 0.185 | 0.200 | 0.015 | 5.353 | 2.436 |
| earth::trot_solid_fast | 30 | 1.000 | 0.147 | 0.171 | 0.024 | 9.114 | 2.343 |
