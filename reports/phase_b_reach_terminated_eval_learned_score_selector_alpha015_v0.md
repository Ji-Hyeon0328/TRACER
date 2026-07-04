# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_ppo_learned_score_selector_alpha015_v0`
- Result root: `artifacts/phase_b_learned_score_selector_alpha015_eval_v0`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 3 | 0.667 | 0.231 | 0.294 | 0.063 | 6.810 | 2.118 | 5.739 |
| stairs_single | 3 | 1.000 | 0.062 | 0.539 | 0.476 | 9.625 | 1.630 | 7.261 |
| tracer_sponge_firm_flat | 3 | 0.667 | 0.211 | 2.713 | 2.501 | 6.733 | -1.537 | 3.487 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 3 | 0.667 | 0.231 | 0.294 | 0.063 | 6.810 | 2.118 |
| stairs_single::trot_solid_fast | 3 | 1.000 | 0.062 | 0.539 | 0.476 | 9.625 | 1.630 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 3 | 0.667 | 0.211 | 2.713 | 2.501 | 6.733 | -1.537 |
