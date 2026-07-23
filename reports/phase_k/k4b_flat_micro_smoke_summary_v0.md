# TRACER Phase-K4 Flat Micro-Action Candidates Summary v0

- log root: `logs/phase_k/k4b_flat_micro_smoke_20260723_162724`
- summary csv: `datasets/phase_k/k4b_flat_micro_smoke_summary_v0.csv`
- timeout guard was enabled per rollout.

## Status counts

| mode | status | count |
|---|---|---:|
| baseline | ok | 1 |
| flat_noop | ok | 1 |

## OK-run group summary

| mode | n | goals | max_abs_y mean | max_abs_y min | max_abs_y max | mean_abs_y mean | j7 accepted mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 1 | 0.373435 | 0.373435 | 0.373435 | 0.167873 | 0.00 |
| flat_noop | 1 | 1 | 0.478062 | 0.478062 | 0.478062 | 0.164146 | 477.00 |

## Per-run results

| status | mode | idx | goal | final_x | final_y | max_abs_y | mean_abs_y | j7 accepted | flat acc | projected | empirical | failsafe |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ok | baseline | 1 | True | 8.132175 | 0.175472 | 0.373435 | 0.167873 | 0 | NA | 0 | 0 | 0 |
| ok | flat_noop | 1 | True | 8.107035 | 0.147606 | 0.478062 | 0.164146 | 477 | 477/478 | 477 | 1788 | 8 |

## Interpretation guide

- A candidate is promising only if it reaches all goals and improves both max_abs_y and mean_abs_y over baseline.
- Timeout rows are infrastructure failures, not locomotion failures.
- If no candidate beats baseline, keep active routing as scaffold/debug and move to learned/data-backed selector design.
