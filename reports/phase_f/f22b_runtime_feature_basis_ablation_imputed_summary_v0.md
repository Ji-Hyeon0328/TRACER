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
| 1 | ram_ref | 6 | 0.439738 | 0.922557 | p060 | p015 |
| 2 | d7_beta_only | 12 | 0.442037 | 0.907106 | p060 | p015 |
| 3 | score_proxy | 5 | 0.442128 | 0.902811 | p060 | p015 |
| 4 | lateral_runtime | 8 | 0.447732 | 0.934446 | p060 | p015 |
| 5 | all_runtime_safe | 29 | 0.477710 | 0.959696 | p060 | p015 |
| 6 | all_runtime_safe_plus_lateral_basis | 47 | 0.514549 | 0.997107 | p060 | p015 |

## Per-condition LOTO for top suites

| suite | heldout | target beta | pred beta | LOTO L1 |
|---|---|---|---|---:|
| ram_ref | clean | (0.3807, 0.2387, 0.3807) | (0.4249, 0.2819, 0.2932) | 0.174908 |
| ram_ref | m015 | (0.5301, 0.2194, 0.2505) | (0.4001, 0.3121, 0.2878) | 0.287161 |
| ram_ref | m030 | (0.3610, 0.1145, 0.5245) | (0.4266, 0.2997, 0.2736) | 0.501657 |
| ram_ref | m060 | (0.6204, 0.1594, 0.2202) | (0.3809, 0.2938, 0.3253) | 0.478927 |
| ram_ref | p015 | (0.3885, 0.2666, 0.3449) | (0.4230, 0.2772, 0.2997) | 0.093541 |
| ram_ref | p030 | (0.6232, 0.3430, 0.0338) | (0.3835, 0.2730, 0.3435) | 0.619414 |
| ram_ref | p060 | (0.0287, 0.6033, 0.3680) | (0.4874, 0.2265, 0.2861) | 0.922557 |
| d7_beta_only | clean | (0.3807, 0.2387, 0.3807) | (0.4278, 0.2873, 0.2849) | 0.191623 |
| d7_beta_only | m015 | (0.5301, 0.2194, 0.2505) | (0.3949, 0.2822, 0.3229) | 0.274515 |
| d7_beta_only | m030 | (0.3610, 0.1145, 0.5245) | (0.4203, 0.3049, 0.2748) | 0.499215 |
| d7_beta_only | m060 | (0.6204, 0.1594, 0.2202) | (0.3764, 0.2962, 0.3274) | 0.487878 |
| d7_beta_only | p015 | (0.3885, 0.2666, 0.3449) | (0.4271, 0.2822, 0.2908) | 0.108750 |
| d7_beta_only | p030 | (0.6232, 0.3430, 0.0338) | (0.3824, 0.2712, 0.3464) | 0.625173 |
| d7_beta_only | p060 | (0.0287, 0.6033, 0.3680) | (0.4822, 0.2205, 0.2974) | 0.907106 |
| score_proxy | clean | (0.3807, 0.2387, 0.3807) | (0.4265, 0.2871, 0.2865) | 0.188515 |
| score_proxy | m015 | (0.5301, 0.2194, 0.2505) | (0.3889, 0.2797, 0.3314) | 0.292431 |
| score_proxy | m030 | (0.3610, 0.1145, 0.5245) | (0.4243, 0.3025, 0.2732) | 0.502509 |
| score_proxy | m060 | (0.6204, 0.1594, 0.2202) | (0.3898, 0.2884, 0.3219) | 0.464512 |
| score_proxy | p015 | (0.3885, 0.2666, 0.3449) | (0.4280, 0.2832, 0.2888) | 0.116846 |
| score_proxy | p030 | (0.6232, 0.3430, 0.0338) | (0.3801, 0.2725, 0.3474) | 0.627270 |
| score_proxy | p060 | (0.0287, 0.6033, 0.3680) | (0.4801, 0.2225, 0.2973) | 0.902811 |

## Safe interpretation

If all_runtime_safe_plus_lateral_basis substantially improves over all_runtime_safe, nonlinear lateral interactions are useful. If not, the remaining bottleneck is missing RAM/context information rather than simply the linear ridge basis.
