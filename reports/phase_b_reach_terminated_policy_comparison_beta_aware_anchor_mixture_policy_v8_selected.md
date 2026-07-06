# Phase-B Reach-Terminated Policy Comparison

Main metric: reach-terminated traversal success. Final distance and R_s are auxiliary frozen-controller stability diagnostics.

| world | policy | n | reach_rate | mean_min_dist | mean_final_dist | post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | fixed_trot_cautious | 3 | 0.333 | 0.282 | 0.373 | 0.091 | 4.882 | 2.004 | 4.189 |
| earth | fixed_trot_mid | 3 | 0.667 | 0.158 | 0.257 | 0.099 | 7.647 | 2.139 | 6.412 |
| earth | fixed_trot_soft_mid_clear | 3 | 0.000 | 0.261 | 0.261 | 0.000 | 3.793 | 2.378 | 3.382 |
| earth | fixed_trot_solid_fast | 3 | 1.000 | 0.146 | 0.243 | 0.097 | 9.121 | 2.414 | 7.622 |
| earth | tracer_beta_aware_anchor_mixture_policy_v8_selected | 5 | 1.000 | 0.149 | 0.210 | 0.061 | 9.104 | 2.331 | 7.610 |
| earth | tracer_beta_aware_candidate_policy_v7_selected | 5 | 1.000 | 0.147 | 0.199 | 0.052 | 9.115 | 2.379 | 7.626 |
| earth | tracer_beta_ram_candidate_policy_v3_robust_clean | 3 | 1.000 | 0.148 | 0.219 | 0.071 | 9.111 | 2.442 | 7.633 |
| earth | tracer_beta_ram_candidate_policy_v5_robust_clean_long | 5 | 1.000 | 0.147 | 0.192 | 0.045 | 9.114 | 2.350 | 7.620 |
| earth | tracer_proxy_ppo_argmax_v1 | 3 | 1.000 | 0.148 | 0.187 | 0.039 | 9.110 | 2.514 | 7.643 |
| stairs_single | fixed_trot_cautious | 3 | 0.667 | 0.132 | 0.286 | 0.154 | 7.720 | 1.826 | 5.958 |
| stairs_single | fixed_trot_mid | 3 | 1.000 | 0.059 | 0.182 | 0.123 | 9.647 | 1.994 | 7.375 |
| stairs_single | fixed_trot_soft_mid_clear | 3 | 0.333 | 0.306 | 0.511 | 0.205 | 4.591 | 1.645 | 3.665 |
| stairs_single | fixed_trot_solid_fast | 3 | 0.667 | 0.203 | 0.720 | 0.517 | 6.900 | 1.409 | 5.247 |
| stairs_single | tracer_beta_aware_anchor_mixture_policy_v8_selected | 5 | 0.800 | 0.115 | 0.630 | 0.515 | 8.451 | 1.306 | 6.337 |
| stairs_single | tracer_beta_aware_candidate_policy_v7_selected | 5 | 1.000 | 0.070 | 0.280 | 0.210 | 9.577 | 2.033 | 7.320 |
| stairs_single | tracer_beta_ram_candidate_policy_v3_robust_clean | 3 | 1.000 | 0.045 | 0.353 | 0.308 | 9.727 | 2.165 | 7.464 |
| stairs_single | tracer_beta_ram_candidate_policy_v5_robust_clean_long | 5 | 1.000 | 0.094 | 0.626 | 0.532 | 9.437 | 1.453 | 7.084 |
| stairs_single | tracer_proxy_ppo_argmax_v1 | 3 | 0.667 | 0.150 | 1.008 | 0.858 | 7.605 | 0.756 | 5.599 |
| tracer_sponge_firm_flat | fixed_trot_cautious | 3 | 0.333 | 0.293 | 0.851 | 0.557 | 4.610 | 0.877 | 3.049 |
| tracer_sponge_firm_flat | fixed_trot_mid | 3 | 0.000 | 0.214 | 2.163 | 1.949 | 4.343 | -1.334 | 2.104 |
| tracer_sponge_firm_flat | fixed_trot_soft_mid_clear | 3 | 0.000 | 0.472 | 0.763 | 0.291 | 1.326 | 1.402 | 1.267 |
| tracer_sponge_firm_flat | fixed_trot_solid_fast | 3 | 0.667 | 0.122 | 3.986 | 3.864 | 7.912 | -3.657 | 3.417 |
| tracer_sponge_firm_flat | tracer_anchor_mixture_v8_sponge_top3 | 20 | 0.250 | 0.219 | 2.815 | 2.596 | 5.148 | -2.199 | 2.301 |
| tracer_sponge_firm_flat | tracer_anchor_mixture_v8b_sponge_sweep | 40 | 0.700 | 0.138 | 3.501 | 3.363 | 7.747 | -2.946 | 3.598 |
| tracer_sponge_firm_flat | tracer_anchor_mixture_v8c_phase_scheduled | 25 | 0.560 | 0.193 | 3.328 | 3.135 | 6.625 | -2.649 | 3.032 |
| tracer_sponge_firm_flat | tracer_beta_aware_anchor_mixture_policy_v8_selected | 5 | 0.400 | 0.272 | 2.952 | 2.680 | 5.219 | -2.162 | 2.353 |
| tracer_sponge_firm_flat | tracer_beta_aware_candidate_policy_v7_selected | 5 | 0.200 | 0.267 | 2.232 | 1.964 | 4.397 | -1.108 | 2.236 |
| tracer_sponge_firm_flat | tracer_beta_ram_candidate_policy_v3_robust_clean | 3 | 0.333 | 0.335 | 0.427 | 0.092 | 4.025 | 1.650 | 2.967 |
| tracer_sponge_firm_flat | tracer_beta_ram_candidate_policy_v5_robust_clean_long | 5 | 0.800 | 0.184 | 3.268 | 3.084 | 7.689 | -2.499 | 3.720 |
| tracer_sponge_firm_flat | tracer_proxy_ppo_argmax_v1 | 3 | 1.000 | 0.067 | 3.822 | 3.755 | 9.596 | -3.043 | 4.681 |
