# TRACER Phase-G3 Split Diagnostic v0

This compares leave-one-condition-out vs leave-one-rollout-out validation on G1 enriched runtime features.

- input: `datasets/phase_g/g1_enriched_runtime_feature_table_v0.csv`
- output csv: `datasets/phase_g/g3_split_diagnostic_v0.csv`
- rollouts: `21`
- alpha: `1.0`

## Split summary

| suite | split | mean L1 | max L1 | worst | best |
|---|---|---:|---:|---|---|
| online_safe_g3 | leave_one_condition | 0.608253 | 1.043200 | p060 | p015 |
| online_safe_g3 | leave_one_rollout | 0.663609 | 1.637041 | p060_r2 | p015_r2 |
| ram_ref_context_g3 | leave_one_condition | 0.565804 | 0.918083 | p060 | clean |
| ram_ref_context_g3 | leave_one_rollout | 0.522932 | 1.582195 | p060 | m060_r2 |
| tracking_energy_proxy_g3 | leave_one_condition | 0.522679 | 1.245095 | p060 | p015 |
| tracking_energy_proxy_g3 | leave_one_rollout | 0.474714 | 1.600423 | p060_r2 | p015 |
| true_metric_upper_bound_g3_leaky | leave_one_condition | 0.428244 | 1.012411 | p060 | clean |
| true_metric_upper_bound_g3_leaky | leave_one_rollout | 0.363935 | 0.991802 | p060_r1 | clean_r2 |

## Interpretation guide

- If leave-one-rollout is much better than leave-one-condition, the bottleneck is condition coverage / extrapolation.
- If leave-one-rollout is also poor, the features do not explain rollout-level variability.
- If even the leaky upper bound is poor under leave-one-condition, collecting anchor conditions near p060/p030 is more valuable than adding another regressor.
