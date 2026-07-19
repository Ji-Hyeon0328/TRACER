# TRACER Phase-D5 Learned Selector v0

- model_type: `ridge_regression_multi_output`
- source_dataset: `datasets/phase_d5/phase_d5_shadow_dataset_20260718_215059_dataset_v0.csv`
- out_json: `models/phase_d5/d5_learned_selector_ridge_from_shadow_dataset_20260718_215059_v0.json`
- num_samples: `736`
- num_train: `588`
- num_test: `148`
- alpha: `0.0001`

## Overall metrics

| target | train_rmse | train_mae | test_rmse | test_mae |
|---|---:|---:|---:|---:|
| target_vx | 0.01027298 | 0.00130000 | 0.01446675 | 0.00241865 |
| target_yaw_rate | 0.00057958 | 0.00034405 | 0.00085007 | 0.00060061 |
| target_body_h | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| target_clearance | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| target_enable | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |

## Context-wise test metrics

| context | n | vx_rmse | yaw_rmse | body_h_rmse | clearance_rmse | enable_rmse |
|---|---:|---:|---:|---:|---:|---:|
| downslope | 48 | 0.00027858 | 0.00025063 | 0.00000000 | 0.00000000 | 0.00000000 |
| goal_flat | 62 | 0.02234905 | 0.00126093 | 0.00000000 | 0.00000000 | 0.00000000 |
| rough | 38 | 0.00027990 | 0.00037542 | 0.00000000 | 0.00000000 | 0.00000000 |
