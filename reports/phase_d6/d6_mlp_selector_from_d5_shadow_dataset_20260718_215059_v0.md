# TRACER Phase-D6 MLP Selector v0

- model_type: `supervised_mlp_selector`
- source_dataset: `datasets/phase_d5/phase_d5_shadow_dataset_20260718_215059_dataset_v0.csv`
- out_pt: `models/phase_d6/d6_mlp_selector_from_d5_shadow_dataset_20260718_215059_v0.pt`
- out_json: `models/phase_d6/d6_mlp_selector_from_d5_shadow_dataset_20260718_215059_v0.json`
- out_pred_csv: `reports/phase_d6/d6_mlp_selector_predictions_20260718_215059_v0.csv`
- num_samples: `736`
- num_train: `588`
- num_test: `148`
- hidden_sizes: `[64, 64]`
- epochs: `2500`
- best_train_normalized_mse: `0.004640613682568073`

## Metrics

| target | train_rmse | train_mae | test_rmse | test_mae | all_rmse | all_mae |
|---|---:|---:|---:|---:|---:|---:|
| target_vx | 0.01010728 | 0.00145589 | 0.01483967 | 0.00199924 | 0.01122039 | 0.00156515 |
| target_yaw_rate | 0.00028108 | 0.00005922 | 0.00011293 | 0.00004690 | 0.00025628 | 0.00005674 |
| target_body_h | 0.00108642 | 0.00068764 | 0.00092358 | 0.00067140 | 0.00105569 | 0.00068437 |
| target_clearance | 0.00001524 | 0.00001065 | 0.00001514 | 0.00001109 | 0.00001522 | 0.00001074 |
| target_enable | 0.00126972 | 0.00092764 | 0.00130294 | 0.00098530 | 0.00127647 | 0.00093923 |

## Context-wise test metrics

| context | n | vx_rmse | yaw_rmse | body_h_rmse | clearance_rmse | enable_rmse |
|---|---:|---:|---:|---:|---:|---:|
| downslope | 30 | 0.00023010 | 0.00004492 | 0.00085596 | 0.00001098 | 0.00076305 |
| flat | 23 | 0.00012985 | 0.00001341 | 0.00143302 | 0.00000331 | 0.00060309 |
| goal_flat | 43 | 0.02752861 | 0.00020105 | 0.00033488 | 0.00001005 | 0.00127644 |
| rough | 27 | 0.00009882 | 0.00003962 | 0.00127423 | 0.00002516 | 0.00185236 |
| upslope | 25 | 0.00036167 | 0.00004120 | 0.00057869 | 0.00001857 | 0.00158390 |
