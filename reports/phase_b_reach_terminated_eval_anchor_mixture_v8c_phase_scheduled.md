# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_anchor_mixture_v8c_phase_scheduled`
- Result root: `artifacts/phase_b_anchor_mixture_v8c_phase_scheduled_eval`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat | 25 | 0.560 | 0.193 | 3.328 | 3.135 | 6.625 | -2.649 | 3.032 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat::sponge_v8b_reach_bias_fast | 5 | 0.000 | 0.293 | 2.923 | 2.630 | 3.412 | -2.289 |
| tracer_sponge_firm_flat::sponge_v8c_far_fast_early_stop | 5 | 0.600 | 0.212 | 2.788 | 2.576 | 6.531 | -1.763 |
| tracer_sponge_firm_flat::sponge_v8c_mid_fast_early_stop | 5 | 0.800 | 0.178 | 3.821 | 3.643 | 7.802 | -3.451 |
| tracer_sponge_firm_flat::sponge_v8c_soft_stop | 5 | 0.600 | 0.180 | 3.323 | 3.143 | 6.959 | -2.832 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 5 | 0.800 | 0.099 | 3.783 | 3.684 | 8.423 | -2.911 |
