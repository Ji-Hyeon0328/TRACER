# TRACER Phase-L1 K6 Seed Dataset Summary v0

- source: `datasets/phase_k/k6_selector_candidate_outcome_dataset_v0.csv`
- output: `datasets/phase_l/l1_selector_dataset_seed_from_k6_v0.csv`
- schema version: `phase_l1_v0`
- row granularity: `rollout_candidate_outcome`
- rows: 50

## Decision labels

- `empirical`: 16
- `reject`: 28
- `shadow_only`: 6

## Source phases

- `K1`: 20
- `K4C`: 30

## Candidate modes

- `baseline`: 16
- `flat_clear03`: 6
- `flat_noop`: 6
- `flat_slow03`: 6
- `flat_slow03_clear03`: 6
- `j19_profile`: 10

## Interpretation

- This is a rollout-level candidate outcome seed dataset.
- It is not yet a state/context-conditioned selector training dataset.
- K6 does not contain explicit context, beta, RAM rho/sigma, or numerical candidate action fields.
- Missing input features are intentionally left empty rather than inferred from candidate names.
- `reference` is normalized to the canonical `empirical` label.
- `shadow_only` is treated as near-miss supervision, not a positive deployment label.
- No `candidate_positive_later` samples are created in this step.
- Active runtime remains empirical/default.

## Next step

Build a broader Phase-L dataset that logs state/context/objective/action features
at segment or candidate-decision time, then join those inputs with rollout outcomes.
