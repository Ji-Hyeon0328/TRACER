# Phase-B Invalid Gait Calibration v1b

- Input: `data/phase_b_invalid_gait_labels_v1/ablation_data_v1_stairs_invalid_gait_labels.jsonl`
- Episodes: `60`
- Hard invalid: `6`
- Warning: `1`

## By world

| world | n | hard_invalid_rate | warning_rate | missing_state_rate |
|---|---:|---:|---:|---:|
| stairs_single | 60 | 0.100 | 0.017 | 0.000 |

## By action

| world::action | n | hard_invalid_rate | warning_rate | hard reasons | warning reasons |
|---|---:|---:|---:|---|---|
| stairs_single::trot_mid | 30 | 0.167 | 0.033 | large_roll_or_pitch:5 | metric_suspicious_low_progress_stable:1 |
| stairs_single::trot_solid_fast | 30 | 0.033 | 0.000 | large_roll_or_pitch:1 |  |
