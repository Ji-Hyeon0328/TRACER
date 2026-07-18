# TRACER Phase-D4 Context Table Bank Comparison v0

- bank_manifest: `reports/phase_d4_context_table_bank_20260718_143619_summaries.tsv`
- best_name: `rough_clearance`
- best_score: `96.0498`
- best_hold_vx: `0.0250`

## Best action table

```
flat:0.210,0.320,0.045;start_flat:0.210,0.320,0.045;upslope:0.210,0.320,0.045;rough:0.2050,0.320,0.055;downslope:0.2025,0.320,0.045;goal_flat:0.2025,0.320,0.045;unknown:0.2025,0.320,0.045
```

| rank | name | score | success_rate | goal_rate | first_goal_time_mean | final_goal_error_mean | hold_drift_mean | max_abs_y_mean | mean_abs_y_mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | rough_clearance | 96.0498 | 1.0000 | 1.0000 | 178.6667 | 0.0380 | 0.0720 | 0.4267 | 0.1600 |
| 2 | conservative_down | 95.4215 | 1.0000 | 1.0000 | 182.6667 | 0.0800 | 0.0313 | 0.4067 | 0.1471 |
| 3 | smooth_all | 95.1306 | 1.0000 | 1.0000 | 185.3333 | 0.1153 | 0.0127 | 0.3377 | 0.1403 |
| 4 | baseline_hold025 | 93.5063 | 1.0000 | 1.0000 | 178.6667 | 0.1673 | 0.0067 | 0.4380 | 0.2099 |
| 5 | aggressive_flat_rough | 91.6038 | 1.0000 | 1.0000 | 175.6667 | 0.1730 | 0.0393 | 0.5937 | 0.2913 |
