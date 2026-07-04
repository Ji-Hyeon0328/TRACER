# Phase-B Reach-Terminated Policy Comparison

Main metric: reach-terminated traversal success. Final distance and R_s are auxiliary frozen-controller stability diagnostics.

| world | policy | n | reach_rate | mean_min_dist | mean_final_dist | post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earth | fixed_trot_cautious | 3 | 0.333 | 0.282 | 0.373 | 0.091 | 4.882 | 2.004 | 4.189 |
| earth | fixed_trot_mid | 3 | 0.667 | 0.158 | 0.257 | 0.099 | 7.647 | 2.139 | 6.412 |
| earth | fixed_trot_soft_mid_clear | 3 | 0.000 | 0.261 | 0.261 | 0.000 | 3.793 | 2.378 | 3.382 |
| earth | fixed_trot_solid_fast | 3 | 1.000 | 0.146 | 0.243 | 0.097 | 9.121 | 2.414 | 7.622 |
| earth | tracer_ppo_with_objective_selector_ram_proxy_v0 | 3 | 0.667 | 0.244 | 0.304 | 0.060 | 6.664 | 1.877 | 5.581 |
| earth | tracer_proxy_ppo_argmax_v1 | 3 | 1.000 | 0.148 | 0.187 | 0.039 | 9.110 | 2.514 | 7.643 |
| stairs_single | fixed_trot_cautious | 3 | 0.667 | 0.132 | 0.286 | 0.154 | 7.720 | 1.826 | 5.958 |
| stairs_single | fixed_trot_mid | 3 | 1.000 | 0.059 | 0.182 | 0.123 | 9.647 | 1.994 | 7.375 |
| stairs_single | fixed_trot_soft_mid_clear | 3 | 0.333 | 0.306 | 0.511 | 0.205 | 4.591 | 1.645 | 3.665 |
| stairs_single | fixed_trot_solid_fast | 3 | 0.667 | 0.203 | 0.720 | 0.517 | 6.900 | 1.409 | 5.247 |
| stairs_single | tracer_ppo_with_objective_selector_ram_proxy_v0 | 3 | 0.000 | 0.333 | 0.984 | 0.652 | 2.961 | 0.829 | 2.288 |
| stairs_single | tracer_proxy_ppo_argmax_v1 | 3 | 0.667 | 0.150 | 1.008 | 0.858 | 7.605 | 0.756 | 5.599 |
| tracer_sponge_firm_flat | fixed_trot_cautious | 3 | 0.333 | 0.293 | 0.851 | 0.557 | 4.610 | 0.877 | 3.049 |
| tracer_sponge_firm_flat | fixed_trot_mid | 3 | 0.000 | 0.214 | 2.163 | 1.949 | 4.343 | -1.334 | 2.104 |
| tracer_sponge_firm_flat | fixed_trot_soft_mid_clear | 3 | 0.000 | 0.472 | 0.763 | 0.291 | 1.326 | 1.402 | 1.267 |
| tracer_sponge_firm_flat | fixed_trot_solid_fast | 3 | 0.667 | 0.122 | 3.986 | 3.864 | 7.912 | -3.657 | 3.417 |
| tracer_sponge_firm_flat | tracer_ppo_with_objective_selector_ram_proxy_v0 | 3 | 0.000 | 0.339 | 2.302 | 1.963 | 2.882 | -1.522 | 1.175 |
| tracer_sponge_firm_flat | tracer_proxy_ppo_argmax_v1 | 3 | 1.000 | 0.067 | 3.822 | 3.755 | 9.596 | -3.043 | 4.681 |
