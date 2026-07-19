# TRACER Phase-D6 Ridge vs MLP Selector Comparison v0

- ridge_json: `models/phase_d5/d5_learned_selector_ridge_from_shadow_dataset_20260718_215059_v0.json`
- mlp_json: `models/phase_d6/d6_mlp_selector_from_d5_shadow_dataset_20260718_215059_v0.json`

| target | ridge_test_rmse | mlp_test_rmse | ridge_test_mae | mlp_test_mae |
|---|---:|---:|---:|---:|
| target_vx | 0.01446675 | 0.01483967 | 0.00241865 | 0.00199924 |
| target_yaw_rate | 0.00085007 | 0.00011293 | 0.00060061 | 0.00004690 |
| target_body_h | 0.00000000 | 0.00092358 | 0.00000000 | 0.00067140 |
| target_clearance | 0.00000000 | 0.00001514 | 0.00000000 | 0.00001109 |
| target_enable | 0.00000000 | 0.00130294 | 0.00000000 | 0.00098530 |
