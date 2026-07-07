# Phase-B Invalid Gait Detection v1

- Root: `artifacts/phase_b_ablation_data_v1/stairs`
- Episodes: `60`
- Invalid: `6`
- Missing state: `0`

## By world

| world | n | invalid_rate | missing_state_rate |
|---|---:|---:|---:|
| stairs_single | 60 | 0.100 | 0.000 |

## By action

| world::action | n | invalid_rate | missing_state_rate | top reasons |
|---|---:|---:|---:|---|
| stairs_single::trot_mid | 30 | 0.167 | 0.000 | large_roll_or_pitch:5, metric_suspicious_low_progress_stable:1 |
| stairs_single::trot_solid_fast | 30 | 0.033 | 0.000 | large_roll_or_pitch:1 |
