# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_sponge_phase_switch_v1c`
- Result root: `artifacts/phase_b_sponge_phase_switch_v1c_eval`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat | 60 | 0.217 | 0.346 | 0.837 | 0.491 | 3.584 | 1.050 | 2.498 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat::sponge_slow_high_clear | 10 | 0.200 | 0.336 | 1.086 | 0.750 | 3.658 | 0.582 |
| tracer_sponge_firm_flat::sponge_v1c_conservative_reach_hold | 10 | 0.300 | 0.340 | 0.746 | 0.406 | 3.955 | 1.254 |
| tracer_sponge_firm_flat::sponge_v1c_reach_bias_to_hold | 10 | 0.400 | 0.280 | 0.761 | 0.480 | 5.023 | 1.136 |
| tracer_sponge_firm_flat::sponge_v1c_reach_stabilized_to_hold | 10 | 0.000 | 0.403 | 0.864 | 0.462 | 2.135 | 1.016 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 10 | 0.200 | 0.360 | 0.789 | 0.429 | 3.316 | 1.209 |
| tracer_sponge_firm_flat::sponge_v8b_reach_stabilized | 10 | 0.200 | 0.354 | 0.775 | 0.421 | 3.415 | 1.102 |
