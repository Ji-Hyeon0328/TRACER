# TRACER Phase-L6E Rough Rollout Evidence v0

- source evidence: `datasets/phase_l/l6d_fresh_rough_clearance_validity_window_evidence_v1.csv`
- source outcome: `datasets/phase_l/l6d_fresh_rough_clearance_validity_outcome_index_v0.csv`
- input rough windows: 27
- output rollout rows: 24
- rough completion threshold: x >= 5.950
- continuous rough metrics preserve each rollout as one statistical unit.

## Mode counts

- `baseline`: 6
- `rough_clear_high05`: 6
- `rough_clear_low05`: 6
- `rough_noop`: 6

## Reliability

| mode | rough completed | goal reached |
|---|---:|---:|
| baseline | 6/6 | 6/6 |
| rough_clear_high05 | 6/6 | 6/6 |
| rough_clear_low05 | 6/6 | 5/6 |
| rough_noop | 5/6 | 5/6 |

## Outcome classes

| mode | class | count |
|---|---|---:|
| baseline | goal_success | 6 |
| rough_clear_high05 | goal_success | 6 |
| rough_clear_low05 | goal_failure_after_rough | 1 |
| rough_clear_low05 | goal_success | 5 |
| rough_noop | goal_success | 5 |
| rough_noop | rough_incomplete | 1 |

## Fragmented source rollouts

| mode | repeat | windows | elapsed span | observed duration | coverage | outcome |
|---|---:|---:|---:|---:|---:|---|
| rough_noop | 2 | 2 | 44.600 | 44.500 | 0.998 | goal_success |
| rough_noop | 3 | 2 | 140.000 | 133.601 | 0.954 | rough_incomplete |
| rough_noop | 5 | 2 | 43.000 | 42.900 | 0.998 | goal_success |

## Per-rollout rough evidence

| repeat | mode | windows | elapsed | progress | rate | mean |y| | max |y| | rough complete | goal | outcome |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | baseline | 1 | 44.100 | 1.990879 | 0.045145 | 0.073022 | 0.169381 | 1 | 1 | goal_success |
| 1 | rough_clear_high05 | 1 | 44.700 | 1.997301 | 0.044683 | 0.169012 | 0.238692 | 1 | 1 | goal_success |
| 1 | rough_clear_low05 | 1 | 42.700 | 1.992658 | 0.046666 | 0.311468 | 0.340888 | 1 | 1 | goal_success |
| 1 | rough_noop | 1 | 43.299 | 1.992708 | 0.046022 | 0.042029 | 0.093659 | 1 | 1 | goal_success |
| 2 | baseline | 1 | 42.800 | 1.996856 | 0.046656 | 0.349099 | 0.428164 | 1 | 1 | goal_success |
| 2 | rough_clear_high05 | 1 | 45.599 | 1.992572 | 0.043697 | 0.268177 | 0.361529 | 1 | 1 | goal_success |
| 2 | rough_clear_low05 | 1 | 41.900 | 1.996247 | 0.047643 | 0.073829 | 0.170602 | 1 | 1 | goal_success |
| 2 | rough_noop | 2 | 44.600 | 1.997339 | 0.044783 | 0.050549 | 0.091559 | 1 | 1 | goal_success |
| 3 | baseline | 1 | 43.200 | 1.995539 | 0.046193 | 0.361793 | 0.449502 | 1 | 1 | goal_success |
| 3 | rough_clear_high05 | 1 | 44.499 | 1.995966 | 0.044854 | 0.365820 | 0.466060 | 1 | 1 | goal_success |
| 3 | rough_clear_low05 | 1 | 43.400 | 1.993869 | 0.045942 | 0.041763 | 0.096159 | 1 | 1 | goal_success |
| 3 | rough_noop | 2 | 140.000 | 0.120441 | 0.000860 | 0.447632 | 1.021802 | 0 | 0 | rough_incomplete |
| 4 | baseline | 1 | 42.100 | 1.992490 | 0.047328 | 0.112966 | 0.204764 | 1 | 1 | goal_success |
| 4 | rough_clear_high05 | 1 | 43.600 | 1.993781 | 0.045728 | 0.082904 | 0.194362 | 1 | 1 | goal_success |
| 4 | rough_clear_low05 | 1 | 43.900 | 1.993091 | 0.045401 | 0.056745 | 0.120119 | 1 | 1 | goal_success |
| 4 | rough_noop | 1 | 42.400 | 1.993048 | 0.047006 | 0.071452 | 0.166264 | 1 | 1 | goal_success |
| 5 | baseline | 1 | 43.100 | 1.997870 | 0.046354 | 0.146609 | 0.214669 | 1 | 1 | goal_success |
| 5 | rough_clear_high05 | 1 | 46.300 | 1.993473 | 0.043056 | 0.320225 | 0.397163 | 1 | 1 | goal_success |
| 5 | rough_clear_low05 | 1 | 109.500 | 1.995586 | 0.018224 | 0.147503 | 0.343669 | 1 | 0 | goal_failure_after_rough |
| 5 | rough_noop | 2 | 43.000 | 1.999054 | 0.046490 | 0.234920 | 0.314107 | 1 | 1 | goal_success |
| 6 | baseline | 1 | 40.100 | 1.994790 | 0.049745 | 0.659897 | 0.833110 | 1 | 1 | goal_success |
| 6 | rough_clear_high05 | 1 | 45.500 | 1.995034 | 0.043847 | 0.357782 | 0.440553 | 1 | 1 | goal_success |
| 6 | rough_clear_low05 | 1 | 43.600 | 1.996504 | 0.045792 | 0.297395 | 0.320121 | 1 | 1 | goal_success |
| 6 | rough_noop | 1 | 45.200 | 1.996233 | 0.044165 | 0.359001 | 0.459873 | 1 | 1 | goal_success |

## Interpretation

- `rough_elapsed_span_s` includes gaps between fragmented evidence windows.
- `rough_observed_duration_s` is the sum of observed window durations.
- Reliability outcomes include all six rollouts per mode.
- Continuous quality comparisons must state whether they are conditional on rough completion.
- No candidate is promoted by this aggregation step.
