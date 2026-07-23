# TRACER Phase-K3 Conservative Risk Rule Summary v0

- source segment csv: `datasets/phase_k/k2b_segment_metrics_from_k1_corrected_v0.csv`
- output rule json: `configs/phase_k/k3_conservative_risk_rule_from_k2b_v0.json`

## Global decision

- deploy performance policy: `False`
- recommended runtime mode: `empirical_default_active_shadow_only`
- actually active contexts: `['flat']`
- allowed active contexts: `[]`
- blocked active contexts: `['flat']`

## Context-level decision

| context | Δ max_abs_y | Δ mean_abs_y | Δ abs_delta_y | j7 accepted mean | j7 projected mean | decision | reason |
|---|---:|---:|---:|---:|---:|---|---|
| downslope | -0.036353 | -0.043224 | +0.024258 | 0.00 | 0.00 | block | active_degrades_segment_metrics |
| flat | +0.027321 | +0.019119 | +0.027245 | 448.22 | 448.22 | block | active_degrades_segment_metrics |
| goal_flat | -0.012250 | -0.000178 | -0.024641 | 0.00 | 0.00 | protected_empirical | nonworse_but_no_active_projection_in_this_context |
| rough | +0.022600 | +0.029542 | +0.000737 | 0.00 | 0.00 | block | active_degrades_segment_metrics |
| upslope | +0.061311 | +0.041361 | +0.049593 | 0.00 | 0.00 | block | active_degrades_segment_metrics |

## Interpretation

- K3 uses a conservative non-worsening rule.
- A context is allowed only if active metrics are non-worse than baseline and active projection actually occurred there.
- If the actually active context is degraded, the profile is not promoted as a deployable performance policy.
- The J19 profile can still be kept as a scaffold/debug active-routing profile.
