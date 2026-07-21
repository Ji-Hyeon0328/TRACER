# TRACER Phase-E1 Rollout Objective Metrics v1

- fixes v0 reset_y parsing using reset_y_offset_v0.log
- computes path_max_abs_y/path_mean_abs_y directly from D7 y trajectory

- output_csv: `datasets/phase_e/e1_rollout_objective_metrics_v2.csv`
- num_rollouts: `13`
- manifests:
  - `reports/phase_d4_context_meta_repeat_20260720_155704_manifest.tsv`
  - `reports/phase_d4_context_meta_repeat_20260720_162902_manifest.tsv`
  - `reports/phase_d4_context_meta_repeat_20260720_164714_manifest.tsv`
  - `reports/phase_d4_context_meta_repeat_20260720_180654_manifest.tsv`
  - `reports/phase_d4_context_meta_repeat_20260720_192118_manifest.tsv`
  - `reports/phase_d4_context_meta_repeat_20260720_192832_manifest.tsv`
  - `reports/phase_d4_context_meta_repeat_20260720_194813_manifest.tsv`
  - `reports/phase_d4_context_meta_repeat_20260720_223320_manifest.tsv`
  - `reports/phase_d4_context_meta_repeat_20260720_224250_manifest.tsv`

## Rows

| idx | trial | reset_y | learned vx/yaw/clr | success | goal | out_lane | final_x | final_y | path_max_abs_y | path_mean_abs_y | moving_accept | motion | stability | energy | effort |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | +0.00 | True/True/True | 1 | 1 | 0 | 7.979 | -0.140 | 0.196 | 0.080 | 0.994 | 0.575 | 0.963 | 0.562 | 0.438 |
| 1 | 2 | +0.00 | True/True/True | 1 | 1 | 0 | 8.180 | +0.138 | 0.364 | 0.150 | 0.992 | 0.574 | 0.924 | 0.560 | 0.440 |
| 2 | 3 | +0.00 | True/True/True | 1 | 1 | 0 | 7.829 | -0.058 | 0.417 | 0.179 | 0.994 | 0.575 | 0.923 | 0.560 | 0.440 |
| 3 | 1 | -0.30 | True/True/True | 1 | 1 | 0 | 8.188 | +0.694 | 0.694 | 0.222 | 0.929 | 0.572 | 0.898 | 0.556 | 0.444 |
| 4 | 1 | +0.30 | True/True/True | 1 | 1 | 0 | 8.138 | +0.082 | 0.086 | 0.034 | 0.995 | 0.575 | 0.977 | 0.563 | 0.437 |
| 5 | 1 | -0.30 | True/False/True | 1 | 1 | 0 | 8.133 | +0.664 | 0.667 | 0.286 | 0.960 | 0.577 | 0.873 | 0.563 | 0.437 |
| 6 | 1 | -0.30 | False/True/True | 1 | 1 | 0 | 8.248 | +0.474 | 0.505 | 0.244 | 0.994 | 0.578 | 0.877 | 0.565 | 0.435 |
| 7 | 1 | -0.30 | False/False/True | 1 | 1 | 0 | 8.004 | +0.183 | 0.199 | 0.081 | 0.994 | 0.576 | 0.962 | 0.564 | 0.436 |
| 8 | 1 | -0.30 | False/False/True | 1 | 1 | 0 | 8.066 | -0.259 | 0.321 | 0.144 | 0.993 | 0.571 | 0.932 | 0.561 | 0.439 |
| 9 | 2 | -0.30 | False/False/True | 1 | 1 | 0 | 8.211 | -0.503 | 0.518 | 0.209 | 0.993 | 0.570 | 0.889 | 0.557 | 0.443 |
| 10 | 3 | -0.30 | False/False/True | 1 | 1 | 0 | 8.051 | -0.443 | 0.501 | 0.331 | 0.994 | 0.567 | 0.862 | 0.551 | 0.449 |
| 11 | 1 | +0.30 | False/False/True | 1 | 1 | 0 | 8.293 | -0.008 | 0.327 | 0.128 | 0.995 | 0.573 | 0.926 | 0.563 | 0.437 |
| 12 | 1 | +0.00 | False/False/True | 1 | 1 | 0 | 8.148 | -0.120 | 0.353 | 0.150 | 0.994 | 0.577 | 0.921 | 0.558 | 0.442 |
