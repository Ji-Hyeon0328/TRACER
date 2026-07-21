# TRACER Phase-G3 Split Diagnostic v0

This compares leave-one-condition-out vs leave-one-rollout-out validation on G1 enriched runtime features.

- input: `datasets/phase_g/g1_enriched_runtime_feature_table_v0.csv`
- output csv: `datasets/phase_g/g3_split_diagnostic_v0.csv`
- rollouts: `22`
- alpha: `1.0`

## Split summary

| suite | split | mean L1 | max L1 | worst | best |
|---|---|---:|---:|---|---|
| online_safe_g3 | leave_one_condition | 0.433056 | 0.913592 | p060 | m060 |
| online_safe_g3 | leave_one_rollout | 0.456907 | 1.554420 | p060_r2 | clean |
| ram_ref_context_g3 | leave_one_condition | 0.387122 | 0.727846 | p060 | p030 |
| ram_ref_context_g3 | leave_one_rollout | 0.362354 | 1.165774 | p060 | m060_r2 |
| tracking_energy_proxy_g3 | leave_one_condition | 0.366466 | 1.065904 | p060 | clean |
| tracking_energy_proxy_g3 | leave_one_rollout | 0.317566 | 1.325894 | m015_r2 | clean |
| true_metric_upper_bound_g3_leaky | leave_one_condition | 0.312125 | 0.890352 | p060 | clean |
| true_metric_upper_bound_g3_leaky | leave_one_rollout | 0.265575 | 1.086469 | m015_r2 | p030 |

## Interpretation guide

- If leave-one-rollout is much better than leave-one-condition, the bottleneck is condition coverage / extrapolation.
- If leave-one-rollout is also poor, the features do not explain rollout-level variability.
- If even the leaky upper bound is poor under leave-one-condition, collecting anchor conditions near p060/p030 is more valuable than adding another regressor.
