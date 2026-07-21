# TRACER Phase-H0 Objective Selector Training Table v0

This builds a rollout-level supervised pretraining table for the Objective Selector beta model.

- input: `datasets/phase_g/g1_enriched_runtime_feature_table_v0.csv`
- output csv: `datasets/phase_h/h0_objective_selector_training_table_v0.csv`
- feature manifest: `datasets/phase_h/h0_objective_selector_feature_manifest_v0.json`
- rows: `22`
- total feature columns: `57`
- deployable-pretrain features: `54`
- diagnostic-only features: `3`

## Rows by base condition

| base_tag | rows |
|---|---:|
| clean | 3 |
| m015 | 3 |
| m030 | 3 |
| m060 | 3 |
| p015 | 3 |
| p030 | 3 |
| p045 | 1 |
| p060 | 3 |

## Target beta summary

| target | mean | std | min | max |
|---|---:|---:|---:|---:|
| beta_motion_target | 0.357367 | 0.143023 | 0.026538 | 0.571137 |
| beta_stability_target | 0.308464 | 0.110738 | 0.180938 | 0.557289 |
| beta_energy_target | 0.334168 | 0.103279 | 0.045788 | 0.487809 |

## Dominant beta counts

| dominant target | rows |
|---|---:|
| energy | 3 |
| motion | 16 |
| stability | 3 |

## Feature policy

- Old D7 `pred/raw/prior/actual_beta_*` columns are excluded as inputs.
- Target beta, true-seed labels, and dominant labels are excluded as inputs.
- Direct `rollout_true_metric_*` columns are excluded; derived tracking/contact/energy ratios are kept as diagnostic online-window candidates.
- `reset_y_*` features are kept only in the diagnostic-only group because reset offset is an artificial experimental condition.

## Safe interpretation

The beta targets are robust true-metric teacher seeds for supervised Objective Selector pretraining. They are not yet final IRL labels. H1 should train both deployable-pretrain and diagnostic variants and report them separately.
