# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_ppo_learned_score_selector_alpha055_stairs_v0`
- Result root: `artifacts/phase_b_learned_score_selector_alpha055_stairs_eval_v0`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stairs_single | 3 | 0.667 | 0.111 | 0.280 | 0.169 | 7.999 | 1.861 | 6.157 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| stairs_single::trot_mid | 3 | 0.667 | 0.111 | 0.280 | 0.169 | 7.999 | 1.861 |
