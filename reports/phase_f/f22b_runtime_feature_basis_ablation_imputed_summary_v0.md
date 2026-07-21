# TRACER Phase-F22b Runtime Feature Basis Ablation with Imputation v0

This repeats F22 with train-mean feature imputation so all_runtime_safe and nonlinear lateral-basis suites are evaluated instead of being skipped by strict row filtering.

- input: `datasets/phase_f/f19_d7_robust_beta_calibration_dataset_v0.csv`
- F20 model feature source: `models/phase_f/f20_robust_runtime_safe_beta_calibrator_ridge_v0.json`
- output csv: `datasets/phase_f/f22b_runtime_feature_basis_ablation_imputed_v0.csv`
- alpha: `1.0`
- F20 feature count: `36`

## Suite summary

| rank | suite | n_features | mean LOTO L1 | max LOTO L1 | worst condition | best condition |
|---:|---|---:|---:|---:|---|---|
| 1 | d7_beta_only | 12 | 0.319220 | 0.762651 | p060 | p015 |
| 2 | score_proxy | 5 | 0.319688 | 0.762764 | p060 | p015 |
| 3 | ram_ref | 6 | 0.320068 | 0.770295 | p060 | p015 |
| 4 | lateral_runtime | 8 | 0.321070 | 0.769391 | p060 | p015 |
| 5 | all_runtime_safe | 29 | 0.339593 | 0.780762 | p060 | p015 |
| 6 | all_runtime_safe_plus_lateral_basis | 47 | 0.357985 | 0.796922 | p060 | p015 |

## Per-condition LOTO for top suites

| suite | heldout | target beta | pred beta | LOTO L1 |
|---|---|---|---|---:|
| d7_beta_only | clean | (0.3655, 0.2691, 0.3655) | (0.3582, 0.3163, 0.3255) | 0.098801 |
| d7_beta_only | m015 | (0.4287, 0.2704, 0.3009) | (0.3401, 0.3106, 0.3493) | 0.180727 |
| d7_beta_only | m030 | (0.3313, 0.1809, 0.4878) | (0.3517, 0.3292, 0.3190) | 0.345119 |
| d7_beta_only | m060 | (0.4489, 0.2333, 0.3178) | (0.3360, 0.3191, 0.3449) | 0.227766 |
| d7_beta_only | p015 | (0.3672, 0.2869, 0.3460) | (0.3586, 0.3134, 0.3280) | 0.059671 |
| d7_beta_only | p030 | (0.4623, 0.3365, 0.2012) | (0.3452, 0.3078, 0.3470) | 0.291578 |
| d7_beta_only | p045 | (0.5711, 0.3831, 0.0458) | (0.3524, 0.3081, 0.3395) | 0.587447 |
| d7_beta_only | p060 | (0.0265, 0.5573, 0.4162) | (0.4079, 0.2665, 0.3256) | 0.762651 |
| score_proxy | clean | (0.3655, 0.2691, 0.3655) | (0.3574, 0.3162, 0.3264) | 0.098046 |
| score_proxy | m015 | (0.4287, 0.2704, 0.3009) | (0.3345, 0.3089, 0.3566) | 0.197627 |
| score_proxy | m030 | (0.3313, 0.1809, 0.4878) | (0.3552, 0.3273, 0.3175) | 0.344718 |
| score_proxy | m060 | (0.4489, 0.2333, 0.3178) | (0.3464, 0.3129, 0.3407) | 0.209146 |
| score_proxy | p015 | (0.3672, 0.2869, 0.3460) | (0.3597, 0.3139, 0.3264) | 0.062813 |
| score_proxy | p030 | (0.4623, 0.3365, 0.2012) | (0.3439, 0.3086, 0.3475) | 0.293888 |
| score_proxy | p045 | (0.5711, 0.3831, 0.0458) | (0.3493, 0.3107, 0.3400) | 0.588506 |
| score_proxy | p060 | (0.0265, 0.5573, 0.4162) | (0.4079, 0.2680, 0.3241) | 0.762764 |
| ram_ref | clean | (0.3655, 0.2691, 0.3655) | (0.3562, 0.3129, 0.3310) | 0.087728 |
| ram_ref | m015 | (0.4287, 0.2704, 0.3009) | (0.3437, 0.3327, 0.3236) | 0.187991 |
| ram_ref | m030 | (0.3313, 0.1809, 0.4878) | (0.3618, 0.3248, 0.3134) | 0.348785 |
| ram_ref | m060 | (0.4489, 0.2333, 0.3178) | (0.3416, 0.3172, 0.3412) | 0.214700 |
| ram_ref | p015 | (0.3672, 0.2869, 0.3460) | (0.3557, 0.3100, 0.3343) | 0.048204 |
| ram_ref | p030 | (0.4623, 0.3365, 0.2012) | (0.3399, 0.3089, 0.3512) | 0.299992 |
| ram_ref | p045 | (0.5711, 0.3831, 0.0458) | (0.3472, 0.3056, 0.3472) | 0.602849 |
| ram_ref | p060 | (0.0265, 0.5573, 0.4162) | (0.4117, 0.2714, 0.3169) | 0.770295 |

## Safe interpretation

If all_runtime_safe_plus_lateral_basis substantially improves over all_runtime_safe, nonlinear lateral interactions are useful. If not, the remaining bottleneck is missing RAM/context information rather than simply the linear ridge basis.
