# TRACER Phase-L2 Window Action Evidence Summary v0

- source K6: `datasets/phase_k/k6_selector_candidate_outcome_dataset_v0.csv`
- output: `datasets/phase_l/l2_window_action_evidence_from_k6_v0.csv`
- output rows: 112
- skipped source rows: 2

## Window types

- `accepted_candidate_exposure`: 32
- `empirical_context_window`: 80

## Evidence roles

- `control_equivalent_noop`: 14
- `near_miss_exposure`: 6
- `reference_anchor`: 80
- `rollout_associated_negative_unpaired`: 12

## Candidate modes

- `baseline`: 80
- `flat_clear03`: 6
- `flat_noop`: 5
- `flat_slow03`: 6
- `flat_slow03_clear03`: 6
- `j19_profile`: 9

## Contexts

- `downslope`: 16
- `flat`: 48
- `goal_flat`: 16
- `rough`: 16
- `upslope`: 16

## Skipped source rows

- `j19_profile`: `missing_d5_log_dir_metadata`
- `flat_noop`: `missing_d5_log_dir_metadata`

## Interpretation

- Rows represent context/action windows, not independent 10 Hz samples.
- Candidate rows require realized J7 projected exposure.
- K6 decisions are preserved only as rollout-level labels.
- `direct_action_label_allowed` remains false for all L2 rows.
- Control-equivalent no-op windows are explicitly separated from causal negative evidence.
- This dataset is an evidence table for later effect estimation and selector training design.
