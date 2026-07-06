# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_sponge_late_switch_v1d`
- Result root: `artifacts/phase_b_sponge_late_switch_v1d_eval`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat | 90 | 0.178 | 0.343 | 0.863 | 0.520 | 3.470 | 1.025 | 2.418 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat::sponge_slow_high_clear | 10 | 0.000 | 0.400 | 0.891 | 0.490 | 2.165 | 1.046 |
| tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_025_013 | 10 | 0.300 | 0.245 | 1.144 | 0.899 | 4.999 | 0.490 |
| tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_030_015 | 10 | 0.100 | 0.421 | 0.887 | 0.466 | 2.310 | 1.124 |
| tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_035_016 | 10 | 0.100 | 0.399 | 0.866 | 0.467 | 2.562 | 1.082 |
| tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_025_013 | 10 | 0.000 | 0.397 | 0.867 | 0.470 | 2.204 | 1.091 |
| tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_030_015 | 10 | 0.100 | 0.371 | 0.686 | 0.314 | 2.905 | 1.300 |
| tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_035_016 | 10 | 0.400 | 0.271 | 0.836 | 0.565 | 5.066 | 0.984 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 10 | 0.500 | 0.236 | 0.699 | 0.463 | 5.877 | 1.142 |
| tracer_sponge_firm_flat::sponge_v8b_reach_stabilized | 10 | 0.100 | 0.351 | 0.893 | 0.543 | 3.144 | 0.961 |
