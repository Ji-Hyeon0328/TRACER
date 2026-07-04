# Phase-B Reach-Terminated Evaluation

- Policy: `fixed_trot_solid_fast`
- Result root: `artifacts/phase_b_fixed_baseline_eval_v0/trot_solid_fast`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 3 | 1.000 | 0.146 | 0.243 | 0.097 | 9.121 | 2.414 | 7.622 |
| stairs_single | 3 | 0.667 | 0.203 | 0.720 | 0.517 | 6.900 | 1.409 | 5.247 |
| tracer_sponge_firm_flat | 3 | 0.667 | 0.122 | 3.986 | 3.864 | 7.912 | -3.657 | 3.417 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_solid_fast | 3 | 1.000 | 0.146 | 0.243 | 0.097 | 9.121 | 2.414 |
| stairs_single::trot_solid_fast | 3 | 0.667 | 0.203 | 0.720 | 0.517 | 6.900 | 1.409 |
| tracer_sponge_firm_flat::trot_solid_fast | 3 | 0.667 | 0.122 | 3.986 | 3.864 | 7.912 | -3.657 |
