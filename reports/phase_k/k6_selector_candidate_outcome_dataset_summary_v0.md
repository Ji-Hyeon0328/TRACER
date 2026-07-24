# TRACER Phase-K6 Selector Dataset Summary v0

- K1 source: `datasets/phase_k/k1_long_baseline_vs_j19_profile_summary_v0.csv`
- K4C source: `datasets/phase_k/k4c_flat_micro_candidates_fixed_summary_v0.csv`
- K3 risk rule: `configs/phase_k/k3_conservative_risk_rule_from_k2b_v0.json`
- K5 decision: `configs/phase_k/k5_flat_micro_candidate_promotion_decision_v0.json`
- output dataset: `datasets/phase_k/k6_selector_candidate_outcome_dataset_v0.csv`

## Group summary

| source | mode | n | ok | goals | active used mean | max_abs_y mean | mean_abs_y mean | risk score mean | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| K1 | baseline | 10 | 10 | 10 | 0.000 | 0.365898 | 0.157337 | 0.444567 | `['reference']` |
| K1 | j19_profile | 10 | 9 | 9 | 0.900 | 0.347311 | 0.165474 | 1.037043 | `['reject']` |
| K4C | baseline | 6 | 6 | 6 | 0.000 | 0.335127 | 0.152324 | 0.411288 | `['reference']` |
| K4C | flat_clear03 | 6 | 6 | 6 | 1.000 | 0.399815 | 0.223024 | 0.511327 | `['reject']` |
| K4C | flat_noop | 6 | 5 | 5 | 0.833 | 0.396019 | 0.174589 | 1.486094 | `['reject']` |
| K4C | flat_slow03 | 6 | 6 | 6 | 1.000 | 0.329978 | 0.157207 | 0.408582 | `['shadow_only']` |
| K4C | flat_slow03_clear03 | 6 | 6 | 6 | 1.000 | 0.387286 | 0.179587 | 0.477080 | `['reject']` |

## Dataset usage

- This dataset is not yet enough to train a strong RL policy.
- It is useful for a conservative selector/risk classifier prototype.
- Positive deploy labels are intentionally absent because K5 promoted no active candidate.
- `flat_slow03` can be kept as a shadow-only near-miss sample.
- The next step is K7: build a conservative selector that predicts `empirical`, `shadow_only`, or `reject` from context/action/outcome features.
