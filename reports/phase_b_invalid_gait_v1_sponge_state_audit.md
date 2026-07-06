# Phase-B Invalid Gait Detection v1

- Root: `artifacts/phase_b_sponge_state_audit_v1`
- Episodes: `20`
- Invalid: `8`
- Missing state: `0`

## By world

| world | n | invalid_rate | missing_state_rate |
|---|---:|---:|---:|
| tracer_sponge_firm_flat | 20 | 0.400 | 0.000 |

## By action

| world::action | n | invalid_rate | missing_state_rate | top reasons |
|---|---:|---:|---:|---|
| tracer_sponge_firm_flat::sponge_slow_high_clear | 5 | 0.600 | 0.000 | low_base_height:3, folded_joint_posture:FR:1 |
| tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_025_013 | 5 | 0.200 | 0.000 | low_base_height:1 |
| tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_035_016 | 5 | 0.400 | 0.000 | low_base_height:2, folded_joint_posture:FL:1, large_roll_or_pitch:1, metric_suspicious_low_progress_stable:1 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 5 | 0.400 | 0.000 | low_base_height:2, metric_suspicious_low_progress_stable:2, folded_joint_posture:FL,FR,RL,RR:1, folded_joint_posture:FR:1, large_roll_or_pitch:1 |
