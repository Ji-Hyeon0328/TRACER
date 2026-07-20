# TRACER Phase-D7 Beta-Conditioned D6 Control Smoke Diagnosis v0

## Summary

- Protected active-beta smoke kept D6 on static beta while D7 published /tracer/objective_beta.
- Beta-conditioned smoke connected D7 /tracer/objective_beta to the D6 MLP selector.
- Rollout still succeeded, but D6 gate acceptance dropped substantially.

## Protected active-beta smoke

- manifest: `reports/phase_d4_context_meta_repeat_20260720_123342_manifest.tsv`
- d5_dir: `/home/kraken/Tracer/TRACER/logs/phase_d5_shadow_20260720_123342_trial_1`
# TRACER Phase-D4 Context-Meta Repeat Summary v0

- manifest: `reports/phase_d4_context_meta_repeat_20260720_123342_manifest.tsv`
- n: `1`
- success_rate: `1.000`
- goal_rate: `1.000`
- startup_failed_rate: `0.000`
- out_lane_rate: `0.000`
- hold_drift_mean: `0.0`

## Individual rollouts

| trial | hold_vx | success | goal | startup_failed | out_lane | final_x | final_y | max_x | max_abs_y | mean_abs_y | hold_drift | final_context |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.025 | True | True | False | False | 8.227 | 0.186 | 8.227 | 0.200 | 0.070 | 0.000 | goal_flat |

```text
[TRACER] csv=/home/kraken/Tracer/TRACER/logs/phase_d5_shadow_20260720_123342_trial_1/gated_selector_dryrun_v0.csv
[TRACER] rows=2423
[TRACER] context_counts:
  flat: 449
  upslope: 414
  rough: 411
  downslope: 489
  goal_flat: 660
[TRACER] accept_count=1754
[TRACER] reject_count=669
[TRACER] accept_rate=0.723896
[TRACER] reject_reasons:
  hold_phase_empirical: 660
  missing_learned: 9
[TRACER] error metrics:
  err_vx: rmse=0.01730865, mae=0.00243667
  err_yaw_rate: rmse=0.00019354, mae=0.00008879
  err_body_h: rmse=0.00627951, mae=0.00465655
  err_clearance: rmse=0.00024707, mae=0.00008273
  err_enable: rmse=0.06094946, mae=0.00381249
[TRACER] accept_rate_by_context:
  flat: 440/449 = 0.979955
  upslope: 414/414 = 1.000000
  rough: 411/411 = 1.000000
  downslope: 489/489 = 1.000000
  goal_flat: 0/660 = 0.000000
[TRACER] phase_accept_rates:
  moving_accept_rate=1754/1763 = 0.994895
  hold_accept_rate=0/660 = 0.000000
```

## Beta-conditioned D6 smoke

- manifest: `reports/phase_d4_context_meta_repeat_20260720_125514_manifest.tsv`
- d5_dir: `/home/kraken/Tracer/TRACER/logs/phase_d5_shadow_20260720_125514_trial_1`
# TRACER Phase-D4 Context-Meta Repeat Summary v0

- manifest: `reports/phase_d4_context_meta_repeat_20260720_125514_manifest.tsv`
- n: `1`
- success_rate: `1.000`
- goal_rate: `1.000`
- startup_failed_rate: `0.000`
- out_lane_rate: `0.000`
- hold_drift_mean: `0.0`

## Individual rollouts

| trial | hold_vx | success | goal | startup_failed | out_lane | final_x | final_y | max_x | max_abs_y | mean_abs_y | hold_drift | final_context |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.025 | True | True | False | False | 8.224 | -0.224 | 8.224 | 0.380 | 0.215 | 0.000 | goal_flat |

```text
[TRACER] csv=/home/kraken/Tracer/TRACER/logs/phase_d5_shadow_20260720_125514_trial_1/gated_selector_dryrun_v0.csv
[TRACER] rows=2420
[TRACER] context_counts:
  flat: 455
  upslope: 410
  rough: 413
  downslope: 464
  goal_flat: 678
[TRACER] accept_count=979
[TRACER] reject_count=1441
[TRACER] accept_rate=0.404545
[TRACER] reject_reasons:
  vx: 757
  hold_phase_empirical: 678
  missing_learned: 6
[TRACER] error metrics:
  err_vx: rmse=0.02162961, mae=0.01372891
  err_yaw_rate: rmse=0.00055361, mae=0.00041485
  err_body_h: rmse=0.01317072, mae=0.01092242
  err_clearance: rmse=0.00017627, mae=0.00010738
  err_enable: rmse=0.05108031, mae=0.00862439
[TRACER] accept_rate_by_context:
  flat: 0/455 = 0.000000
  upslope: 131/410 = 0.319512
  rough: 384/413 = 0.929782
  downslope: 464/464 = 1.000000
  goal_flat: 0/678 = 0.000000
[TRACER] phase_accept_rates:
  moving_accept_rate=979/1742 = 0.561998
  hold_accept_rate=0/678 = 0.000000
```

## Interpretation

- D7 beta topic integration worked.
- D6 MLP consumed D7 beta in the beta-conditioned smoke.
- The rollout remained successful because the D6 gate rejected unsafe learned vx outputs and fell back to empirical reference.
- The large vx rejection count indicates that the current D6 MLP was trained on static/proxy beta inputs and is not yet calibrated for D7 runtime beta inputs.

## Next step

- Build a D7-beta-conditioned D6 training dataset.
- Retrain the D6 MLP using D7 runtime beta as input and safe empirical references as target.
- Repeat beta-conditioned smoke and check whether moving_accept_rate recovers.
