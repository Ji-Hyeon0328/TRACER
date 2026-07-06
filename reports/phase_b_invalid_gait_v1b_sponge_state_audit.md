# Phase-B Invalid Gait Calibration v1b

- Input: `data/phase_b_invalid_gait_labels_v1/sponge_state_audit_v1_invalid_gait_labels.jsonl`
- Episodes: `20`
- Hard invalid: `4`
- Warning: `8`

## By world

| world | n | hard_invalid_rate | warning_rate | missing_state_rate |
|---|---:|---:|---:|---:|
| tracer_sponge_firm_flat | 20 | 0.200 | 0.400 | 0.000 |

## By action

| world::action | n | hard_invalid_rate | warning_rate | hard reasons | warning reasons |
|---|---:|---:|---:|---|---|
| tracer_sponge_firm_flat::sponge_slow_high_clear | 5 | 0.200 | 0.600 | folded_joint_posture:FR:1 | soft_low_base_height:3 |
| tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_025_013 | 5 | 0.000 | 0.200 |  | soft_low_base_height:1 |
| tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_035_016 | 5 | 0.200 | 0.400 | folded_joint_posture:FL:1, large_roll_or_pitch:1, severe_low_base_height:1 | metric_suspicious_low_progress_stable:1, soft_low_base_height:1 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 5 | 0.400 | 0.400 | severe_low_base_height:2, folded_joint_posture:FL,FR,RL,RR:1, folded_joint_posture:FR:1, large_roll_or_pitch:1 | metric_suspicious_low_progress_stable:2 |
