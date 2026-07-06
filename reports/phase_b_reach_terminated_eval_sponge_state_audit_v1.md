# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_sponge_state_audit_v1`
- Result root: `artifacts/phase_b_sponge_state_audit_v1`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat | 20 | 0.700 | 0.147 | 1.692 | 1.544 | 7.617 | -0.546 | 4.359 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat::sponge_slow_high_clear | 5 | 0.600 | 0.112 | 1.952 | 1.840 | 7.575 | -0.994 |
| tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_025_013 | 5 | 0.800 | 0.088 | 2.022 | 1.934 | 8.671 | -1.207 |
| tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_035_016 | 5 | 0.800 | 0.175 | 1.713 | 1.537 | 7.747 | -0.499 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 5 | 0.600 | 0.215 | 1.081 | 0.866 | 6.474 | 0.515 |
