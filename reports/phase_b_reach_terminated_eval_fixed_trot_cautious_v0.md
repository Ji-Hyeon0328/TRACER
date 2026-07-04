# Phase-B Reach-Terminated Evaluation

- Policy: `fixed_trot_cautious`
- Result root: `artifacts/phase_b_fixed_baseline_eval_v0/trot_cautious`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 3 | 0.333 | 0.282 | 0.373 | 0.091 | 4.882 | 2.004 | 4.189 |
| stairs_single | 3 | 0.667 | 0.132 | 0.286 | 0.154 | 7.720 | 1.826 | 5.958 |
| tracer_sponge_firm_flat | 3 | 0.333 | 0.293 | 0.851 | 0.557 | 4.610 | 0.877 | 3.049 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_cautious | 3 | 0.333 | 0.282 | 0.373 | 0.091 | 4.882 | 2.004 |
| stairs_single::trot_cautious | 3 | 0.667 | 0.132 | 0.286 | 0.154 | 7.720 | 1.826 |
| tracer_sponge_firm_flat::trot_cautious | 3 | 0.333 | 0.293 | 0.851 | 0.557 | 4.610 | 0.877 |
