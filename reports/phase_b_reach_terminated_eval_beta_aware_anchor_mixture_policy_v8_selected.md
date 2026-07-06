# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_beta_aware_anchor_mixture_policy_v8_selected`
- Result root: `artifacts/phase_b_beta_aware_candidate_policy_v8_selected_eval`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 5 | 1.000 | 0.149 | 0.210 | 0.061 | 9.104 | 2.331 | 7.610 |
| stairs_single | 5 | 0.800 | 0.115 | 0.630 | 0.515 | 8.451 | 1.306 | 6.337 |
| tracer_sponge_firm_flat | 5 | 0.400 | 0.272 | 2.952 | 2.680 | 5.219 | -2.162 | 2.353 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::trot_mid | 5 | 1.000 | 0.149 | 0.210 | 0.061 | 9.104 | 2.331 |
| stairs_single::trot_solid_fast | 5 | 0.800 | 0.115 | 0.630 | 0.515 | 8.451 | 1.306 |
| tracer_sponge_firm_flat::sponge_v8c_far_fast_early_stop | 5 | 0.400 | 0.272 | 2.952 | 2.680 | 5.219 | -2.162 |
