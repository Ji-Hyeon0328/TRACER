# TRACER Phase-K5 Candidate Promotion Decision v0

- source: `datasets/phase_k/k4c_flat_micro_candidates_fixed_summary_v0.csv`
- output json: `configs/phase_k/k5_flat_micro_candidate_promotion_decision_v0.json`

## Global decision

- deploy active performance policy: `False`
- recommended runtime mode: `empirical_default_active_shadow_only`
- promoted candidates: `[]`

## Candidate decisions

| mode | n ok | goals | max_abs_y mean | mean_abs_y mean | Δ max_abs_y | Δ mean_abs_y | j7 accepted mean | decision | reason |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 6 | 6 | 0.335127 | 0.152324 | +0.000000 | +0.000000 | 0.00 | reference | baseline_reference |
| flat_clear03 | 6 | 6 | 0.399815 | 0.223024 | +0.064689 | +0.070700 | 446.33 | reject | does_not_improve_lateral_metrics |
| flat_noop | 5 | 5 | 0.396019 | 0.174589 | +0.060892 | +0.022265 | 447.00 | reject | does_not_improve_lateral_metrics |
| flat_slow03 | 6 | 6 | 0.329978 | 0.157207 | -0.005148 | +0.004883 | 460.00 | shadow_only | improves_max_abs_y_only_but_worsens_mean_abs_y |
| flat_slow03_clear03 | 6 | 6 | 0.387286 | 0.179587 | +0.052159 | +0.027263 | 464.67 | reject | does_not_improve_lateral_metrics |

## Interpretation

- K4C confirms that the fixed active routing path works.
- No candidate improves both max_abs_y and mean_abs_y relative to baseline.
- flat_slow03 is the closest candidate, but it only slightly improves max_abs_y while worsening mean_abs_y.
- Therefore, no flat micro-action candidate should be promoted as a deployable performance policy.
- Active routing should remain available as scaffold/debug, while empirical/default remains the runtime policy.
