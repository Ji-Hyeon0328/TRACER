# TRACER Phase-D6 MLP Selector v0

- model_type: `supervised_mlp_selector`
- source_dataset: `datasets/phase_d6/d6_d7_beta_conditioned_selector_dataset_v0.csv`
- out_pt: `models/phase_d6/d6_mlp_selector_d7_beta_conditioned_v0.pt`
- out_json: `models/phase_d6/d6_mlp_selector_d7_beta_conditioned_v0.json`
- out_pred_csv: `reports/phase_d6/d6_mlp_selector_d7_beta_conditioned_predictions_v0.csv`
- num_samples: `21797`
- num_train: `17437`
- num_test: `4360`
- hidden_sizes: `[64, 64]`
- epochs: `2500`
- best_train_normalized_mse: `0.3050089478492737`

## Metrics

| target | train_rmse | train_mae | test_rmse | test_mae | all_rmse | all_mae |
|---|---:|---:|---:|---:|---:|---:|
| target_vx | 0.00766844 | 0.00309620 | 0.00722753 | 0.00305827 | 0.00758229 | 0.00308862 |
| target_yaw_rate | 0.00021874 | 0.00009491 | 0.00018945 | 0.00009511 | 0.00021320 | 0.00009495 |
| target_body_h | 0.00417333 | 0.00023192 | 0.00326967 | 0.00021449 | 0.00400891 | 0.00022843 |
| target_clearance | 0.00064888 | 0.00006279 | 0.00046883 | 0.00005574 | 0.00061709 | 0.00006138 |
| target_enable | 0.01304594 | 0.00072203 | 0.01016668 | 0.00066653 | 0.01252309 | 0.00071093 |

## Context-wise test metrics

| context | n | vx_rmse | yaw_rmse | body_h_rmse | clearance_rmse | enable_rmse |
|---|---:|---:|---:|---:|---:|---:|
| downslope | 930 | 0.00239938 | 0.00007565 | 0.00010166 | 0.00017668 | 0.00030128 |
| flat | 810 | 0.00561501 | 0.00015268 | 0.00757394 | 0.00100300 | 0.02355221 |
| goal_flat | 1148 | 0.01266836 | 0.00032297 | 0.00031352 | 0.00006968 | 0.00094022 |
| rough | 748 | 0.00327477 | 0.00011241 | 0.00014562 | 0.00037969 | 0.00050386 |
| upslope | 724 | 0.00252089 | 0.00006523 | 0.00010473 | 0.00003775 | 0.00027203 |
