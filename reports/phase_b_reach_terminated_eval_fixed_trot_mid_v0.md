# Phase-B Reach-Terminated Evaluation

- Policy: `fixed_trot_mid`
- Result root: `artifacts/phase_b_fixed_baseline_eval_v0/trot_mid`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 3 | 0.667 | 0.158 | 0.257 | 0.099 | 7.647 | 2.139 | 6.412 |
| stairs_single | 3 | 1.000 | 0.059 | 0.182 | 0.123 | 9.647 | 1.994 | 7.375 |
| tracer_sponge_firm_flat | 3 | 0.000 | 0.214 | 2.163 | 1.949 | 4.343 | -1.334 | 2.104 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 3 | 0.667 | 0.158 | 0.257 | 0.099 | 7.647 | 2.139 |
| stairs_single::trot_mid | 3 | 1.000 | 0.059 | 0.182 | 0.123 | 9.647 | 1.994 |
| tracer_sponge_firm_flat::trot_mid | 3 | 0.000 | 0.214 | 2.163 | 1.949 | 4.343 | -1.334 |
