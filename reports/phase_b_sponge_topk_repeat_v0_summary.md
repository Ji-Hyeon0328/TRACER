# Phase-B Sponge Top-k Repeat Summary

- Root: `artifacts/phase_b_sponge_topk_repeat_v0/20260703_193658`
- World: `tracer_sponge_firm_flat`

## Interpretation

- Several θ-lite actions are reach-capable on sponge-like flat terrain.
- None of the current θ-lite actions reliably provides post-reach hold stability.
- This suggests the current high-level meta-plan selection can solve traversal/reach partially, but stable hold likely needs an expanded θ interface or low-level adaptive/hold behavior.

## Action Summary

| action | n | reach_rate | mean_min_dist | mean_final_dist | mean_abs_odom_x_delta | best_min_dist | best_final_dist |
|---|---:|---:|---:|---:|---:|---:|---:|
| trot_cautious | 10 | 0.400 | 0.275 | 2.850 | 2.860 | 0.052 | 1.335 |
| trot_high_body_soft | 10 | 0.800 | 0.131 | 3.471 | 3.663 | 0.035 | 2.736 |
| trot_soft_mid_clear | 10 | 0.700 | 0.134 | 3.501 | 3.672 | 0.021 | 2.684 |
| trot_soft_ultraslow | 10 | 0.400 | 0.211 | 2.717 | 2.782 | 0.036 | 0.294 |
| trot_solid_fast | 10 | 0.600 | 0.153 | 3.625 | 3.679 | 0.011 | 2.794 |
