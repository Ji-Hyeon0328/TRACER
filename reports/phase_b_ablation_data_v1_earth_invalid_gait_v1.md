# Phase-B Invalid Gait Detection v1

- Root: `artifacts/phase_b_ablation_data_v1/earth`
- Episodes: `60`
- Invalid: `2`
- Missing state: `0`

## By world

| world | n | invalid_rate | missing_state_rate |
|---|---:|---:|---:|
| earth | 60 | 0.033 | 0.000 |

## By action

| world::action | n | invalid_rate | missing_state_rate | top reasons |
|---|---:|---:|---:|---|
| earth::trot_mid | 30 | 0.067 | 0.000 | low_base_height:2, metric_suspicious_low_progress_stable:2, large_roll_or_pitch:1 |
| earth::trot_solid_fast | 30 | 0.000 | 0.000 |  |
