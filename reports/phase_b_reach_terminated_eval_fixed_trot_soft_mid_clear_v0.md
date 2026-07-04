# Phase-B Reach-Terminated Evaluation

- Policy: `fixed_trot_soft_mid_clear`
- Result root: `artifacts/phase_b_fixed_baseline_eval_v0/trot_soft_mid_clear`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 3 | 0.000 | 0.261 | 0.261 | 0.000 | 3.793 | 2.378 | 3.382 |
| stairs_single | 3 | 0.333 | 0.306 | 0.511 | 0.205 | 4.591 | 1.645 | 3.665 |
| tracer_sponge_firm_flat | 3 | 0.000 | 0.472 | 0.763 | 0.291 | 1.326 | 1.402 | 1.267 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_soft_mid_clear | 3 | 0.000 | 0.261 | 0.261 | 0.000 | 3.793 | 2.378 |
| stairs_single::trot_soft_mid_clear | 3 | 0.333 | 0.306 | 0.511 | 0.205 | 4.591 | 1.645 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 3 | 0.000 | 0.472 | 0.763 | 0.291 | 1.326 | 1.402 |
