# Phase-B Invalid Gait Calibration v1b

- Input: `data/phase_b_invalid_gait_labels_v1/ablation_data_v1_earth_invalid_gait_labels.jsonl`
- Episodes: `60`
- Hard invalid: `2`
- Warning: `2`

## By world

| world | n | hard_invalid_rate | warning_rate | missing_state_rate |
|---|---:|---:|---:|---:|
| earth | 60 | 0.033 | 0.033 | 0.000 |

## By action

| world::action | n | hard_invalid_rate | warning_rate | hard reasons | warning reasons |
|---|---:|---:|---:|---|---|
| earth::trot_mid | 30 | 0.067 | 0.067 | severe_low_base_height:2, large_roll_or_pitch:1 | metric_suspicious_low_progress_stable:2 |
| earth::trot_solid_fast | 30 | 0.000 | 0.000 |  |  |
