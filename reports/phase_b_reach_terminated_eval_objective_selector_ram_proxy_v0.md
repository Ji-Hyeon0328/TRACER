# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_ppo_with_objective_selector_ram_proxy_v0`
- Result root: `artifacts/phase_b_objective_selector_eval_v0`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 3 | 0.667 | 0.244 | 0.304 | 0.060 | 6.664 | 1.877 | 5.581 |
| stairs_single | 3 | 0.000 | 0.333 | 0.984 | 0.652 | 2.961 | 0.829 | 2.288 |
| tracer_sponge_firm_flat | 3 | 0.000 | 0.339 | 2.302 | 1.963 | 2.882 | -1.522 | 1.175 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 3 | 0.667 | 0.244 | 0.304 | 0.060 | 6.664 | 1.877 |
| stairs_single::trot_mid | 3 | 0.000 | 0.333 | 0.984 | 0.652 | 2.961 | 0.829 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 3 | 0.000 | 0.339 | 2.302 | 1.963 | 2.882 | -1.522 |
