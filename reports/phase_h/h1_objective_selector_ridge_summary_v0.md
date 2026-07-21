# TRACER Phase-H1 Objective Selector Ridge v0

This trains ridge-regression Objective Selector beta baselines on the H0 supervised pretraining table.

- input csv: `datasets/phase_h/h0_objective_selector_training_table_v0.csv`
- model: `models/phase_h/h1_objective_selector_ridge_v0.json`
- LOTO eval csv: `datasets/phase_h/h1_objective_selector_loto_eval_v0.csv`
- rows: `22`
- target: robust true-metric teacher seed beta from H0
- postprocess: raw beta prediction projected to probability simplex

## Suite summary

| suite | features | best alpha | mean LOTO L1 | max LOTO L1 | worst heldout | train mean L1 | train max L1 |
|---|---:|---:|---:|---:|---|---:|---:|
| deployable_pretrain | 54 | 100 | 0.323784 | 0.906130 | p060 | 0.215036 | 0.653791 |
| diagnostic_with_reset_y | 57 | 100 | 0.326379 | 0.906191 | p060 | 0.213846 | 0.643942 |

## Best-alpha leave-one-base-tag-out detail

| suite | heldout | n | mean L1 | RMSE motion | RMSE stability | RMSE energy | target mean | pred mean |
|---|---|---:|---:|---:|---:|---:|---|---|
| deployable_pretrain | clean | 3 | 0.100896 | 0.040536 | 0.050145 | 0.031484 | (0.3655, 0.2691, 0.3655) | (0.3514, 0.3110, 0.3376) |
| deployable_pretrain | m015 | 3 | 0.350703 | 0.250914 | 0.236751 | 0.022939 | (0.4287, 0.2704, 0.3009) | (0.2533, 0.4265, 0.3201) |
| deployable_pretrain | m030 | 3 | 0.345688 | 0.058925 | 0.116707 | 0.173540 | (0.3313, 0.1809, 0.4878) | (0.3897, 0.2954, 0.3150) |
| deployable_pretrain | m060 | 3 | 0.195739 | 0.098075 | 0.068533 | 0.036321 | (0.4489, 0.2333, 0.3178) | (0.3510, 0.3005, 0.3484) |
| deployable_pretrain | p015 | 3 | 0.083297 | 0.035244 | 0.041906 | 0.024762 | (0.3672, 0.2869, 0.3460) | (0.3566, 0.3198, 0.3236) |
| deployable_pretrain | p030 | 3 | 0.277336 | 0.114210 | 0.028467 | 0.138775 | (0.4623, 0.3365, 0.2012) | (0.3484, 0.3117, 0.3399) |
| deployable_pretrain | p045 | 1 | 0.576436 | 0.212372 | 0.075846 | 0.288218 | (0.5711, 0.3831, 0.0458) | (0.3588, 0.3072, 0.3340) |
| deployable_pretrain | p060 | 3 | 0.828609 | 0.415273 | 0.277428 | 0.138080 | (0.0265, 0.5573, 0.4162) | (0.4408, 0.2803, 0.2788) |
| diagnostic_with_reset_y | clean | 3 | 0.108533 | 0.037146 | 0.045623 | 0.043164 | (0.3655, 0.2691, 0.3655) | (0.3655, 0.3069, 0.3276) |
| diagnostic_with_reset_y | m015 | 3 | 0.339679 | 0.249693 | 0.236061 | 0.019438 | (0.4287, 0.2704, 0.3009) | (0.2588, 0.4231, 0.3181) |
| diagnostic_with_reset_y | m030 | 3 | 0.345448 | 0.059317 | 0.117910 | 0.173309 | (0.3313, 0.1809, 0.4878) | (0.3890, 0.2959, 0.3151) |
| diagnostic_with_reset_y | m060 | 3 | 0.215793 | 0.108265 | 0.073942 | 0.040516 | (0.4489, 0.2333, 0.3178) | (0.3410, 0.3063, 0.3527) |
| diagnostic_with_reset_y | p015 | 3 | 0.084744 | 0.039368 | 0.045465 | 0.023327 | (0.3672, 0.2869, 0.3460) | (0.3508, 0.3234, 0.3257) |
| diagnostic_with_reset_y | p030 | 3 | 0.276792 | 0.110998 | 0.034288 | 0.138488 | (0.4623, 0.3365, 0.2012) | (0.3523, 0.3081, 0.3396) |
| diagnostic_with_reset_y | p045 | 1 | 0.579586 | 0.215237 | 0.074556 | 0.289793 | (0.5711, 0.3831, 0.0458) | (0.3559, 0.3085, 0.3356) |
| diagnostic_with_reset_y | p060 | 3 | 0.829260 | 0.415583 | 0.278172 | 0.137671 | (0.0265, 0.5573, 0.4162) | (0.4412, 0.2795, 0.2793) |

## Interpretation

- `deployable_pretrain` excludes artificial reset-y diagnostic features.
- `diagnostic_with_reset_y` includes reset-y features and is useful for checking separability, but should not be treated as a deployable robot input.
- This is still supervised pretraining from robust true-metric teacher seeds, not final IRL.
- If LOTO remains weak for p060/p045, the bottleneck is condition coverage and RAM/context representation rather than just the regressor.
