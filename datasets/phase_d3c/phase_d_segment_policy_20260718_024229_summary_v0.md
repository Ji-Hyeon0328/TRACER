# TRACER Phase-D Segment Policy Dataset Summary v0

- manifest: `reports/phase_d_segment_policy_20260718_024229_manifest.tsv`
- goal_x: `8.0`
- lateral_bound: `2.0`
- rollout_csv: `datasets/phase_d3c/phase_d_segment_policy_20260718_024229_rollouts_v0.csv`
- best_label: `m210_2025_confirm`

## Group ranking

| rank | label | n | success_rate | goal_rate | out_lane_rate | startup_failed_rate | score_mean | score_std | time_mean | max_abs_y_mean | mean_abs_y_mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | m210_2025_confirm | 5 | 1.000 | 1.000 | 0.000 | 0.000 | 120.215 | 1.545 | 180.400 | 0.339 | 0.137 |
| 2 | m210_200_confirm | 5 | 1.000 | 1.000 | 0.000 | 0.000 | 117.566 | 5.649 | 182.000 | 0.547 | 0.255 |
| 3 | m205_200_confirm | 5 | 0.400 | 0.400 | 0.000 | 0.000 | 52.757 | 58.653 | 182.000 | 0.515 | 0.262 |

## Policies

| label | schedule |
|---|---|
| m210_2025_confirm | `0,4.000,0.2100,0.3200,0.0450,start;4.000,999,0.2025,0.3200,0.0450,tail` |
| m210_200_confirm | `0,4.000,0.2100,0.3200,0.0450,start;4.000,999,0.2000,0.3200,0.0450,tail` |
| m205_200_confirm | `0,4.000,0.2050,0.3200,0.0450,start;4.000,999,0.2000,0.3200,0.0450,tail` |

## Individual rollouts

| label | trial | success | goal | startup_failed | time | max_abs_y | mean_abs_y | final_x | final_y | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m205_200_confirm | 1 | True | True | False | 188.000 | 0.985 | 0.561 | 7.852 | -0.985 | 111.152 |
| m205_200_confirm | 2 | True | True | False | 176.000 | 0.185 | 0.065 | 7.994 | -0.032 | 122.523 |
| m205_200_confirm | 3 | False | False | False | NA | 0.563 | 0.293 | 7.928 | -0.540 | 8.834 |
| m205_200_confirm | 4 | False | False | False | NA | 0.549 | 0.288 | 7.903 | -0.518 | 8.906 |
| m205_200_confirm | 5 | False | False | False | NA | 0.292 | 0.102 | 7.893 | -0.260 | 12.367 |
| m210_200_confirm | 1 | True | True | False | 184.000 | 0.299 | 0.149 | 7.952 | 0.134 | 120.566 |
| m210_200_confirm | 2 | True | True | False | 176.000 | 0.501 | 0.288 | 7.956 | -0.354 | 118.106 |
| m210_200_confirm | 3 | True | True | False | 182.000 | 1.360 | 0.578 | 7.933 | -1.355 | 107.797 |
| m210_200_confirm | 4 | True | True | False | 186.000 | 0.384 | 0.189 | 7.924 | -0.293 | 119.348 |
| m210_200_confirm | 5 | True | True | False | 182.000 | 0.191 | 0.073 | 7.932 | -0.170 | 122.012 |
| m210_2025_confirm | 1 | True | True | False | 178.000 | 0.430 | 0.221 | 7.877 | -0.321 | 118.792 |
| m210_2025_confirm | 2 | True | True | False | 182.000 | 0.226 | 0.111 | 7.926 | -0.226 | 121.446 |
| m210_2025_confirm | 3 | True | True | False | 176.000 | 0.197 | 0.094 | 7.929 | 0.142 | 122.015 |
| m210_2025_confirm | 4 | True | True | False | 182.000 | 0.352 | 0.091 | 7.922 | -0.352 | 120.273 |
| m210_2025_confirm | 5 | True | True | False | 184.000 | 0.488 | 0.169 | 7.946 | -0.488 | 118.552 |
