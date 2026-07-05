# Phase-B Reach-Terminated Evaluation

- Policy: `tracer_anchor_mixture_v8_sponge_top3`
- Result root: `artifacts/phase_b_anchor_mixture_v8_eval`
- Controller: frozen low-level controller

## Interpretation

- Main success means the robot reached the goal neighborhood at any point during the episode.
- Final distance and hold reward are reported as auxiliary post-reach stability limitations.
- This separates high-level traversal from low-level hold/stabilization.

## By World

| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat | 20 | 0.250 | 0.219 | 2.815 | 2.596 | 5.148 | -2.199 | 2.301 |

## By World and Action

| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| tracer_sponge_firm_flat::sponge_probe_crawlish | 5 | 0.200 | 0.236 | 2.874 | 2.638 | 4.769 | -2.305 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 5 | 0.600 | 0.121 | 3.657 | 3.536 | 7.521 | -3.339 |
| tracer_sponge_firm_flat::sponge_v8_anchor_mix_top3 | 5 | 0.000 | 0.272 | 2.406 | 2.134 | 3.657 | -1.663 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 5 | 0.200 | 0.248 | 2.324 | 2.076 | 4.642 | -1.490 |
