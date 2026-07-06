# Phase-B Reach-Terminated Policy Comparison

Main metric: reach-terminated traversal success. Final distance and R_s are auxiliary frozen-controller stability diagnostics.

| world | policy | n | reach_rate | mean_min_dist | mean_final_dist | post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | tracer_beta_aware_anchor_mixture_policy_v8_selected | 5 | 1.000 | 0.149 | 0.210 | 0.061 | 9.104 | 2.331 | 7.610 |
| earth | tracer_beta_aware_candidate_policy_v7_selected | 5 | 1.000 | 0.147 | 0.199 | 0.052 | 9.115 | 2.379 | 7.626 |
| earth | tracer_beta_ram_candidate_policy_v3_robust_clean | 3 | 1.000 | 0.148 | 0.219 | 0.071 | 9.111 | 2.442 | 7.633 |
| earth | tracer_beta_ram_candidate_policy_v5_robust_clean_long | 5 | 1.000 | 0.147 | 0.192 | 0.045 | 9.114 | 2.350 | 7.620 |
| earth | tracer_theta_residual_actor_v0 | 5 | 1.000 | 0.148 | 0.188 | 0.040 | 9.112 | 2.462 | 7.637 |
| earth | tracer_theta_residual_actor_v1_multimode | 15 | 1.000 | 0.149 | 0.214 | 0.065 | 9.107 | 2.366 | 7.612 |
| earth | tracer_theta_residual_actor_v1b_mode_selected | 5 | 0.800 | 0.217 | 0.364 | 0.146 | 7.499 | 2.089 | 6.287 |
| earth | tracer_theta_residual_actor_v1b_mode_selected_clean | 10 | 1.000 | 0.148 | 0.198 | 0.051 | 9.112 | 2.316 | 7.613 |
| earth | tracer_theta_residual_actor_v1b_multimode | 15 | 0.933 | 0.170 | 0.249 | 0.079 | 8.581 | 2.336 | 7.187 |
| stairs_single | tracer_beta_aware_anchor_mixture_policy_v8_selected | 5 | 0.800 | 0.115 | 0.630 | 0.515 | 8.451 | 1.306 | 6.337 |
| stairs_single | tracer_beta_aware_candidate_policy_v7_selected | 5 | 1.000 | 0.070 | 0.280 | 0.210 | 9.577 | 2.033 | 7.320 |
| stairs_single | tracer_beta_ram_candidate_policy_v3_robust_clean | 3 | 1.000 | 0.045 | 0.353 | 0.308 | 9.727 | 2.165 | 7.464 |
| stairs_single | tracer_beta_ram_candidate_policy_v5_robust_clean_long | 5 | 1.000 | 0.094 | 0.626 | 0.532 | 9.437 | 1.453 | 7.084 |
| stairs_single | tracer_theta_residual_actor_v0 | 5 | 1.000 | 0.057 | 0.604 | 0.547 | 9.657 | 1.505 | 7.256 |
| stairs_single | tracer_theta_residual_actor_v1_multimode | 15 | 1.000 | 0.078 | 0.278 | 0.200 | 9.531 | 1.944 | 7.271 |
| stairs_single | tracer_theta_residual_actor_v1b_mode_selected | 5 | 0.000 | 0.285 | 0.540 | 0.255 | 3.522 | 1.444 | 2.818 |
| stairs_single | tracer_theta_residual_actor_v1b_mode_selected_clean | 10 | 0.900 | 0.078 | 0.399 | 0.320 | 9.109 | 1.675 | 6.898 |
| stairs_single | tracer_theta_residual_actor_v1b_multimode | 15 | 0.933 | 0.091 | 0.511 | 0.420 | 9.177 | 1.592 | 6.934 |
| tracer_sponge_firm_flat | tracer_beta_aware_anchor_mixture_policy_v8_selected | 5 | 0.400 | 0.272 | 2.952 | 2.680 | 5.219 | -2.162 | 2.353 |
| tracer_sponge_firm_flat | tracer_beta_aware_candidate_policy_v7_selected | 5 | 0.200 | 0.267 | 2.232 | 1.964 | 4.397 | -1.108 | 2.236 |
| tracer_sponge_firm_flat | tracer_beta_ram_candidate_policy_v3_robust_clean | 3 | 0.333 | 0.335 | 0.427 | 0.092 | 4.025 | 1.650 | 2.967 |
| tracer_sponge_firm_flat | tracer_beta_ram_candidate_policy_v5_robust_clean_long | 5 | 0.800 | 0.184 | 3.268 | 3.084 | 7.689 | -2.499 | 3.720 |
| tracer_sponge_firm_flat | tracer_theta_residual_actor_v0 | 15 | 0.600 | 0.196 | 3.270 | 3.074 | 6.746 | -2.514 | 3.154 |
| tracer_sponge_firm_flat | tracer_theta_residual_actor_v1_multimode | 15 | 0.467 | 0.238 | 3.133 | 2.895 | 5.811 | -2.466 | 2.603 |
| tracer_sponge_firm_flat | tracer_theta_residual_actor_v1b_mode_selected | 5 | 0.200 | 0.229 | 3.365 | 3.136 | 4.886 | -3.162 | 1.805 |
| tracer_sponge_firm_flat | tracer_theta_residual_actor_v1b_mode_selected_clean | 10 | 0.300 | 0.344 | 0.941 | 0.597 | 3.889 | 0.916 | 2.635 |
| tracer_sponge_firm_flat | tracer_theta_residual_actor_v1b_multimode | 15 | 0.533 | 0.167 | 3.340 | 3.173 | 6.723 | -2.678 | 3.078 |
