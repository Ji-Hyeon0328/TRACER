# Phase-B Reach-Terminated Policy Comparison

Main metric: reach-terminated traversal success. Final distance and R_s are auxiliary frozen-controller stability diagnostics.

| world | policy | n | reach_rate | mean_min_dist | mean_final_dist | post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | fixed_trot_cautious | 3 | 0.333 | 0.282 | 0.373 | 0.091 | 4.882 | 2.004 | 4.189 |
| earth | fixed_trot_mid | 3 | 0.667 | 0.158 | 0.257 | 0.099 | 7.647 | 2.139 | 6.412 |
| earth | fixed_trot_soft_mid_clear | 3 | 0.000 | 0.261 | 0.261 | 0.000 | 3.793 | 2.378 | 3.382 |
| earth | fixed_trot_solid_fast | 3 | 1.000 | 0.146 | 0.243 | 0.097 | 9.121 | 2.414 | 7.622 |
| earth | tracer_beta_aware_candidate_policy_v7_selected | 5 | 1.000 | 0.147 | 0.199 | 0.052 | 9.115 | 2.379 | 7.626 |
| earth | tracer_beta_ram_candidate_policy_v1 | 3 | 1.000 | 0.146 | 0.219 | 0.073 | 9.121 | 2.467 | 7.630 |
| earth | tracer_beta_ram_candidate_policy_v2_robust | 3 | 1.000 | 0.136 | 0.595 | 0.459 | 9.181 | 1.642 | 7.564 |
| earth | tracer_beta_ram_candidate_policy_v3_online_explore_v1 | 6 | 0.667 | 0.220 | 0.437 | 0.218 | 6.938 | 1.909 | 5.799 |
| earth | tracer_beta_ram_candidate_policy_v3_robust_clean | 3 | 1.000 | 0.148 | 0.219 | 0.071 | 9.111 | 2.442 | 7.633 |
| earth | tracer_beta_ram_candidate_policy_v5_robust_clean | 3 | 0.333 | 0.343 | 0.563 | 0.220 | 4.168 | 1.633 | 3.549 |
| earth | tracer_beta_ram_candidate_policy_v5_robust_clean_long | 5 | 1.000 | 0.147 | 0.192 | 0.045 | 9.114 | 2.350 | 7.620 |
| earth | tracer_contextual_ppo_candidate_policy_v1 | 3 | 1.000 | 0.142 | 0.191 | 0.049 | 9.146 | 2.541 | 7.677 |
| earth | tracer_contextual_ppo_candidate_policy_v2_conservative | 3 | 1.000 | 0.148 | 0.198 | 0.049 | 9.109 | 2.307 | 7.609 |
| earth | tracer_proxy_ppo_argmax_v1 | 3 | 1.000 | 0.148 | 0.187 | 0.039 | 9.110 | 2.514 | 7.643 |
| stairs_single | fixed_trot_cautious | 3 | 0.667 | 0.132 | 0.286 | 0.154 | 7.720 | 1.826 | 5.958 |
| stairs_single | fixed_trot_mid | 3 | 1.000 | 0.059 | 0.182 | 0.123 | 9.647 | 1.994 | 7.375 |
| stairs_single | fixed_trot_soft_mid_clear | 3 | 0.333 | 0.306 | 0.511 | 0.205 | 4.591 | 1.645 | 3.665 |
| stairs_single | fixed_trot_solid_fast | 3 | 0.667 | 0.203 | 0.720 | 0.517 | 6.900 | 1.409 | 5.247 |
| stairs_single | tracer_beta_aware_candidate_policy_v7_selected | 5 | 1.000 | 0.070 | 0.280 | 0.210 | 9.577 | 2.033 | 7.320 |
| stairs_single | tracer_beta_ram_candidate_policy_v1 | 3 | 1.000 | 0.080 | 0.442 | 0.363 | 9.520 | 1.813 | 7.229 |
| stairs_single | tracer_beta_ram_candidate_policy_v2_robust | 3 | 1.000 | 0.060 | 0.627 | 0.566 | 9.638 | 1.644 | 7.276 |
| stairs_single | tracer_beta_ram_candidate_policy_v3_online_explore_v1 | 6 | 1.000 | 0.089 | 0.431 | 0.342 | 9.465 | 1.685 | 7.157 |
| stairs_single | tracer_beta_ram_candidate_policy_v3_robust_clean | 3 | 1.000 | 0.045 | 0.353 | 0.308 | 9.727 | 2.165 | 7.464 |
| stairs_single | tracer_beta_ram_candidate_policy_v5_robust_clean | 3 | 0.667 | 0.186 | 0.333 | 0.146 | 7.173 | 1.968 | 5.574 |
| stairs_single | tracer_beta_ram_candidate_policy_v5_robust_clean_long | 5 | 1.000 | 0.094 | 0.626 | 0.532 | 9.437 | 1.453 | 7.084 |
| stairs_single | tracer_contextual_ppo_candidate_policy_v1 | 3 | 0.667 | 0.133 | 0.651 | 0.518 | 7.783 | 1.469 | 5.898 |
| stairs_single | tracer_contextual_ppo_candidate_policy_v2_conservative | 3 | 0.667 | 0.173 | 0.639 | 0.466 | 7.031 | 1.263 | 5.303 |
| stairs_single | tracer_proxy_ppo_argmax_v1 | 3 | 0.667 | 0.150 | 1.008 | 0.858 | 7.605 | 0.756 | 5.599 |
| tracer_sponge_firm_flat | fixed_trot_cautious | 3 | 0.333 | 0.293 | 0.851 | 0.557 | 4.610 | 0.877 | 3.049 |
| tracer_sponge_firm_flat | fixed_trot_mid | 3 | 0.000 | 0.214 | 2.163 | 1.949 | 4.343 | -1.334 | 2.104 |
| tracer_sponge_firm_flat | fixed_trot_soft_mid_clear | 3 | 0.000 | 0.472 | 0.763 | 0.291 | 1.326 | 1.402 | 1.267 |
| tracer_sponge_firm_flat | fixed_trot_solid_fast | 3 | 0.667 | 0.122 | 3.986 | 3.864 | 7.912 | -3.657 | 3.417 |
| tracer_sponge_firm_flat | tracer_beta_aware_candidate_policy_v7_selected | 5 | 0.200 | 0.267 | 2.232 | 1.964 | 4.397 | -1.108 | 2.236 |
| tracer_sponge_firm_flat | tracer_beta_ram_candidate_policy_v1 | 3 | 0.333 | 0.137 | 3.501 | 3.364 | 6.395 | -3.170 | 2.697 |
| tracer_sponge_firm_flat | tracer_beta_ram_candidate_policy_v2_robust | 3 | 0.000 | 0.194 | 3.607 | 3.413 | 4.571 | -3.405 | 1.516 |
| tracer_sponge_firm_flat | tracer_beta_ram_candidate_policy_v3_online_explore_v1 | 6 | 0.667 | 0.140 | 2.678 | 2.537 | 7.561 | -1.928 | 3.842 |
| tracer_sponge_firm_flat | tracer_beta_ram_candidate_policy_v3_robust_clean | 3 | 0.333 | 0.335 | 0.427 | 0.092 | 4.025 | 1.650 | 2.967 |
| tracer_sponge_firm_flat | tracer_beta_ram_candidate_policy_v5_robust_clean | 3 | 0.333 | 0.212 | 2.385 | 2.173 | 5.547 | -1.565 | 2.759 |
| tracer_sponge_firm_flat | tracer_beta_ram_candidate_policy_v5_robust_clean_long | 5 | 0.800 | 0.184 | 3.268 | 3.084 | 7.689 | -2.499 | 3.720 |
| tracer_sponge_firm_flat | tracer_contextual_ppo_candidate_policy_v1 | 3 | 0.000 | 0.388 | 1.309 | 0.921 | 2.313 | 0.237 | 1.436 |
| tracer_sponge_firm_flat | tracer_contextual_ppo_candidate_policy_v2_conservative | 3 | 0.000 | 0.407 | 2.076 | 1.669 | 2.081 | -0.969 | 0.887 |
| tracer_sponge_firm_flat | tracer_proxy_ppo_argmax_v1 | 3 | 1.000 | 0.067 | 3.822 | 3.755 | 9.596 | -3.043 | 4.681 |
