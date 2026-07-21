# TRACER Phase-I1 RAM Ridge Predictor v0

This trains a supervised RAM predictor on the Phase-I0 heuristic RAM teacher dataset.

- input csv: `datasets/phase_i/i0_ram_teacher_dataset_v0.csv`
- manifest: `datasets/phase_i/i0_ram_teacher_manifest_v0.json`
- model: `models/phase_i/i1_ram_ridge_predictor_v0.json`
- eval csv: `datasets/phase_i/i1_ram_ridge_loto_eval_v0.csv`
- rows: `22`
- features: `23`
- targets: `rho_slip_target`, `rho_rough_target`, `sigma_target`

## Summary

| model | best alpha | mean LOTO L1 | max LOTO L1 | worst heldout | train mean L1 | train max L1 |
|---|---:|---:|---:|---|---:|---:|
| mean_baseline | NA | 0.601317 | 1.415095 | p045 | NA | NA |
| ridge | 10 | 0.401097 | 0.898103 | p045 | 0.254751 | 0.802898 |

## Best ridge leave-one-base-tag-out detail

| heldout | n | mean L1 | RMSE slip | RMSE rough | RMSE sigma | target mean | pred mean |
|---|---:|---:|---:|---:|---:|---|---|
| clean | 3 | 0.469892 | 0.160129 | 0.246655 | 0.136783 | (0.3110, 0.1910, 0.3713) | (0.4608, 0.3858, 0.4955) |
| m015 | 3 | 0.484962 | 0.077695 | 0.362045 | 0.131828 | (0.5420, 0.5482, 0.5768) | (0.6016, 0.4689, 0.6996) |
| m030 | 3 | 0.466748 | 0.316821 | 0.085787 | 0.117192 | (0.6874, 0.3446, 0.5569) | (0.4584, 0.4085, 0.4941) |
| m060 | 3 | 0.413098 | 0.097418 | 0.321583 | 0.058757 | (0.5537, 0.6790, 0.5417) | (0.4844, 0.3855, 0.4914) |
| p015 | 3 | 0.315643 | 0.086830 | 0.228119 | 0.089090 | (0.2813, 0.2515, 0.4413) | (0.3544, 0.4202, 0.5052) |
| p030 | 3 | 0.161199 | 0.028494 | 0.132369 | 0.033761 | (0.2684, 0.4881, 0.4718) | (0.2857, 0.3965, 0.4713) |
| p045 | 1 | 0.898103 | 0.009357 | 0.510078 | 0.378668 | (0.2715, 0.8389, 0.8032) | (0.2621, 0.3289, 0.4245) |
| p060 | 3 | 0.330472 | 0.231218 | 0.114964 | 0.113138 | (0.5970, 0.5838, 0.5755) | (0.5071, 0.6374, 0.6495) |

## Interpretation

- This is a RAM supervised pretraining baseline from heuristic teacher seeds.
- It is not yet the final teacher-student RAM trained from proprioceptive prediction error.
- If ridge improves over the mean baseline, the I0 teacher signals are at least partially predictable from runtime-style features.
- Sparse anchors such as p045 should remain high-uncertainty and should not be overfit.
