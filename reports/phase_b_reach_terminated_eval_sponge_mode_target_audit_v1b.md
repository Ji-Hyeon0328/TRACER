# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_sponge_mode_target_audit_v1b`
- Result root: `artifacts/phase_b_theta_residual_actor_v1b_target_audit_sponge`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat | 80 | 0.588 | 0.161 | 3.414 | 3.253 | 7.094 | -2.944 | 3.207 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat::sponge_probe_crawlish | 8 | 0.500 | 0.152 | 3.143 | 2.991 | 6.857 | -2.597 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 8 | 0.750 | 0.141 | 3.857 | 3.717 | 7.896 | -3.365 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 8 | 0.375 | 0.210 | 2.771 | 2.561 | 5.733 | -2.177 |
| tracer_sponge_firm_flat::sponge_v8_anchor_mix_top3 | 8 | 0.500 | 0.186 | 3.375 | 3.189 | 6.534 | -3.133 |
| tracer_sponge_firm_flat::sponge_v8b_probe_reach_fast | 8 | 0.625 | 0.159 | 3.618 | 3.460 | 7.258 | -3.210 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 8 | 0.750 | 0.114 | 3.393 | 3.279 | 8.098 | -2.745 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias_fast | 8 | 0.375 | 0.195 | 3.479 | 3.285 | 5.894 | -3.062 |
| tracer_sponge_firm_flat::sponge_v8b_reach_stabilized | 8 | 0.875 | 0.115 | 3.774 | 3.659 | 8.751 | -3.343 |
| tracer_sponge_firm_flat::sponge_v8c_far_fast_early_stop | 8 | 0.750 | 0.139 | 3.491 | 3.351 | 7.991 | -2.948 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 8 | 0.375 | 0.196 | 3.239 | 3.043 | 5.926 | -2.858 |
