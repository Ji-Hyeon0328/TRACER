# TRACER Phase-K7 Conservative Selector Policy v0

- source dataset: `datasets/phase_k/k6_selector_candidate_outcome_dataset_v0.csv`
- output policy: `configs/phase_k/k7_conservative_selector_policy_v0.json`

## Global decision

- default runtime action: `empirical`
- deploy active performance policy: `False`
- active modes: `[]`
- shadow-only modes: `['flat_slow03']`
- rejected modes: `['flat_clear03', 'flat_noop', 'flat_slow03_clear03', 'j19_profile']`

## Mode rules

| mode | selector action | deploy allowed | n | ok | goals | max_abs_y mean | mean_abs_y mean | risk score mean | j7 accepted mean | reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | empirical | True | 16 | 16 | 16 | 0.354359 | 0.155457 | 0.432087 | 0.00 | baseline_reference |
| flat_clear03 | reject | False | 6 | 6 | 6 | 0.399815 | 0.223024 | 0.511327 | 446.33 | rejected_by_prior_decision |
| flat_noop | reject | False | 6 | 5 | 5 | 0.396019 | 0.174589 | 1.486094 | 372.50 | rejected_by_prior_decision |
| flat_slow03 | shadow_only | False | 6 | 6 | 6 | 0.329978 | 0.157207 | 0.408582 | 460.00 | near_miss_but_not_safe_to_deploy |
| flat_slow03_clear03 | reject | False | 6 | 6 | 6 | 0.387286 | 0.179587 | 0.477080 | 464.67 | rejected_by_prior_decision |
| j19_profile | reject | False | 10 | 9 | 9 | 0.347311 | 0.165474 | 1.037043 | 403.40 | rejected_by_prior_decision |

## Interpretation

- K7 does not promote any active performance policy.
- The runtime-safe choice remains empirical/default control.
- `flat_slow03` is retained only as a shadow-only near-miss candidate.
- This policy is a conservative blocker/selector scaffold, not a learned RL meta-planner.
- Next step should be either Phase-K closure or a broader learned selector dataset with more diverse terrain/action candidates.
