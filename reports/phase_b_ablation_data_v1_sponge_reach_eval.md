# Phase-B Reach-Terminated Evaluation

- Policy: `phase_b_ablation_data_v1_sponge`
- Result root: `artifacts/phase_b_ablation_data_v1/sponge`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat | 120 | 0.592 | 0.148 | 1.904 | 1.755 | 7.203 | -0.872 | 3.998 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat::sponge_slow_high_clear | 30 | 0.567 | 0.156 | 1.869 | 1.713 | 7.064 | -0.878 |
| tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_025_013 | 30 | 0.633 | 0.124 | 2.057 | 1.932 | 7.590 | -1.128 |
| tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_035_016 | 30 | 0.700 | 0.135 | 1.845 | 1.711 | 7.757 | -0.737 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 30 | 0.467 | 0.178 | 1.843 | 1.665 | 6.401 | -0.743 |
