# Phase-B Invalid Gait Calibration v1b

- Input: `data/phase_b_invalid_gait_labels_v1/ablation_data_v1_sponge_invalid_gait_labels.jsonl`
- Episodes: `120`
- Hard invalid: `21`
- Warning: `51`

## By world

| world | n | hard_invalid_rate | warning_rate | missing_state_rate |
|---|---:|---:|---:|---:|
| tracer_sponge_firm_flat | 120 | 0.175 | 0.425 | 0.000 |

## By action

| world::action | n | hard_invalid_rate | warning_rate | hard reasons | warning reasons |
|---|---:|---:|---:|---|---|
| tracer_sponge_firm_flat::sponge_slow_high_clear | 30 | 0.133 | 0.467 | folded_joint_posture:FR:2, severe_low_base_height:2, folded_joint_posture:FL:1 | soft_low_base_height:14 |
| tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_025_013 | 30 | 0.067 | 0.300 | folded_joint_posture:FR:2 | soft_low_base_height:9 |
| tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_035_016 | 30 | 0.233 | 0.467 | folded_joint_posture:FR:3, severe_low_base_height:3, folded_joint_posture:FL:1, folded_joint_posture:FL,FR,RL,RR:1, folded_joint_posture:FL,RL:1, large_roll_or_pitch:1 | soft_low_base_height:12, metric_suspicious_low_progress_stable:2 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 30 | 0.267 | 0.467 | folded_joint_posture:FL:3, folded_joint_posture:FR:3, folded_joint_posture:FR,RR:1, large_roll_or_pitch:1, severe_low_base_height:1 | soft_low_base_height:14 |
