# Phase-B Reach-Terminated Evaluation

- Policy: `phase_b_common_action_bank_pilot_v1`
- Result root: `artifacts/phase_b_common_action_bank_pilot_v1`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | 60 | 0.183 | 0.295 | 0.301 | 0.006 | 4.124 | 2.473 | 3.654 |
| stairs_single | 60 | 0.333 | 0.269 | 0.383 | 0.114 | 4.923 | 1.842 | 3.943 |
| tracer_rough_low | 60 | 0.217 | 0.301 | 0.310 | 0.009 | 4.187 | 2.463 | 3.530 |
| tracer_rough_mid | 60 | 0.217 | 0.291 | 0.296 | 0.005 | 4.303 | 2.547 | 3.634 |
| tracer_slippery_flat | 60 | 0.000 | 0.466 | 0.633 | 0.167 | 1.397 | 1.466 | 1.315 |
| tracer_sponge_firm_flat | 60 | 0.700 | 0.123 | 1.955 | 1.833 | 7.919 | -0.983 | 4.382 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| earth::sponge_slow_high_clear | 10 | 0.000 | 0.437 | 0.439 | 0.002 | 1.741 | 2.477 |
| earth::sponge_v1d_bias_late_hold_025_013 | 10 | 0.000 | 0.248 | 0.249 | 0.000 | 3.945 | 2.575 |
| earth::sponge_v1d_stabilized_late_hold_035_016 | 10 | 0.000 | 0.337 | 0.341 | 0.003 | 2.903 | 2.527 |
| earth::sponge_v8b_reach_bias | 10 | 0.000 | 0.429 | 0.433 | 0.004 | 1.835 | 2.427 |
| earth::trot_mid | 10 | 0.100 | 0.175 | 0.175 | 0.000 | 5.208 | 2.365 |
| earth::trot_solid_fast | 10 | 1.000 | 0.147 | 0.172 | 0.025 | 9.115 | 2.464 |
| stairs_single::sponge_slow_high_clear | 10 | 0.000 | 0.426 | 0.467 | 0.041 | 1.865 | 1.913 |
| stairs_single::sponge_v1d_bias_late_hold_025_013 | 10 | 0.200 | 0.207 | 0.310 | 0.103 | 5.155 | 1.808 |
| stairs_single::sponge_v1d_stabilized_late_hold_035_016 | 10 | 0.100 | 0.326 | 0.365 | 0.039 | 3.435 | 1.900 |
| stairs_single::sponge_v8b_reach_bias | 10 | 0.000 | 0.426 | 0.459 | 0.034 | 1.870 | 1.960 |
| stairs_single::trot_mid | 10 | 0.800 | 0.128 | 0.287 | 0.159 | 8.229 | 1.839 |
| stairs_single::trot_solid_fast | 10 | 0.900 | 0.101 | 0.410 | 0.309 | 8.983 | 1.631 |
| tracer_rough_low::sponge_slow_high_clear | 10 | 0.000 | 0.433 | 0.438 | 0.006 | 1.789 | 2.499 |
| tracer_rough_low::sponge_v1d_bias_late_hold_025_013 | 10 | 0.000 | 0.242 | 0.242 | 0.000 | 4.020 | 2.618 |
| tracer_rough_low::sponge_v1d_stabilized_late_hold_035_016 | 10 | 0.000 | 0.342 | 0.345 | 0.003 | 2.850 | 2.537 |
| tracer_rough_low::sponge_v8b_reach_bias | 10 | 0.000 | 0.424 | 0.430 | 0.006 | 1.891 | 2.449 |
| tracer_rough_low::trot_mid | 10 | 0.500 | 0.156 | 0.158 | 0.001 | 7.020 | 2.438 |
| tracer_rough_low::trot_solid_fast | 10 | 0.800 | 0.212 | 0.248 | 0.036 | 7.549 | 2.240 |
| tracer_rough_mid::sponge_slow_high_clear | 10 | 0.000 | 0.437 | 0.442 | 0.005 | 1.732 | 2.540 |
| tracer_rough_mid::sponge_v1d_bias_late_hold_025_013 | 10 | 0.000 | 0.242 | 0.243 | 0.000 | 4.018 | 2.648 |
| tracer_rough_mid::sponge_v1d_stabilized_late_hold_035_016 | 10 | 0.000 | 0.338 | 0.340 | 0.002 | 2.899 | 2.528 |
| tracer_rough_mid::sponge_v8b_reach_bias | 10 | 0.000 | 0.426 | 0.434 | 0.008 | 1.867 | 2.501 |
| tracer_rough_mid::trot_mid | 10 | 0.300 | 0.164 | 0.164 | 0.000 | 6.126 | 2.526 |
| tracer_rough_mid::trot_solid_fast | 10 | 1.000 | 0.136 | 0.150 | 0.014 | 9.179 | 2.538 |
| tracer_slippery_flat::sponge_slow_high_clear | 10 | 0.000 | 0.484 | 0.607 | 0.123 | 1.186 | 1.523 |
| tracer_slippery_flat::sponge_v1d_bias_late_hold_025_013 | 10 | 0.000 | 0.457 | 0.577 | 0.121 | 1.507 | 1.535 |
| tracer_slippery_flat::sponge_v1d_stabilized_late_hold_035_016 | 10 | 0.000 | 0.479 | 0.696 | 0.217 | 1.247 | 1.385 |
| tracer_slippery_flat::sponge_v8b_reach_bias | 10 | 0.000 | 0.436 | 0.595 | 0.159 | 1.748 | 1.494 |
| tracer_slippery_flat::trot_mid | 10 | 0.000 | 0.475 | 0.641 | 0.166 | 1.291 | 1.462 |
| tracer_slippery_flat::trot_solid_fast | 10 | 0.000 | 0.465 | 0.680 | 0.215 | 1.403 | 1.394 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 10 | 0.700 | 0.127 | 1.974 | 1.847 | 7.918 | -1.074 |
| tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_025_013 | 10 | 0.600 | 0.123 | 1.718 | 1.595 | 7.426 | -0.511 |
| tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_035_016 | 10 | 0.700 | 0.131 | 2.018 | 1.887 | 7.905 | -1.113 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 10 | 0.900 | 0.118 | 1.877 | 1.759 | 8.737 | -0.756 |
| tracer_sponge_firm_flat::trot_mid | 10 | 0.600 | 0.133 | 2.087 | 1.954 | 7.448 | -1.340 |
| tracer_sponge_firm_flat::trot_solid_fast | 10 | 0.700 | 0.103 | 2.059 | 1.956 | 8.081 | -1.106 |
