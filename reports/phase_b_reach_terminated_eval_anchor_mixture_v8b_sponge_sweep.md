# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_anchor_mixture_v8b_sponge_sweep`
- Result root: `artifacts/phase_b_anchor_mixture_v8b_sweep_eval`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat | 40 | 0.700 | 0.138 | 3.501 | 3.363 | 7.747 | -2.946 | 3.598 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat::sponge_reach_then_brake | 5 | 0.600 | 0.128 | 3.738 | 3.609 | 7.493 | -3.302 |
| tracer_sponge_firm_flat::sponge_v8_anchor_mix_top3 | 5 | 0.800 | 0.160 | 3.474 | 3.314 | 7.983 | -2.983 |
| tracer_sponge_firm_flat::sponge_v8b_probe_reach_fast | 5 | 0.800 | 0.098 | 3.851 | 3.753 | 8.567 | -3.245 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 5 | 0.600 | 0.144 | 3.140 | 2.996 | 7.256 | -2.582 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias_fast | 5 | 1.000 | 0.052 | 3.987 | 3.936 | 9.689 | -3.306 |
| tracer_sponge_firm_flat::sponge_v8b_reach_stabilized | 5 | 0.800 | 0.110 | 3.783 | 3.673 | 8.477 | -3.444 |
| tracer_sponge_firm_flat::sponge_v8b_soft_reach | 5 | 0.400 | 0.209 | 3.193 | 2.984 | 5.818 | -2.705 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 5 | 0.600 | 0.202 | 2.840 | 2.638 | 6.693 | -2.000 |
