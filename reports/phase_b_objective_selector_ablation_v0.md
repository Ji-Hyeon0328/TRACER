# Phase-B Objective Selector Ablation Report

- Raw policy: `tracer_proxy_ppo_argmax_v1`
- Selector policy: `tracer_ppo_with_objective_selector_ram_proxy_v0`

This report compares the raw high-level PPO policy against the Objective Selector/RAM proxy v0.

## Conclusion

Objective Selector/RAM proxy v0 degraded performance relative to the raw PPO policy. Treat this as a negative ablation.

| world | raw reach | selector reach | Δreach | raw reward | selector reward | Δreward | ΔR_v | ΔR_s | degraded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| earth | 1.000 | 0.667 | -0.333 | 7.643 | 5.581 | -2.062 | -2.446 | -0.636 | True |
| stairs_single | 0.667 | 0.000 | -0.667 | 5.599 | 2.288 | -3.311 | -4.644 | 0.073 | True |
| tracer_sponge_firm_flat | 1.000 | 0.000 | -1.000 | 4.681 | 1.175 | -3.506 | -6.714 | 1.520 | True |

## Interpretation

- The current proxy is not a learned Objective Selector/RAM module.
- It applies terrain-conditioned hand-coded logit bias, which can override useful PPO preferences.
- The next selector should use empirical robustness, speed, stability, energy proxy, and uncertainty rather than a simple bias.
