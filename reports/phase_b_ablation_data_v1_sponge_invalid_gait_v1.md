# Phase-B Invalid Gait Detection v1

- Root: `artifacts/phase_b_ablation_data_v1/sponge`
- Episodes: `120`
- Invalid: `55`
- Missing state: `0`

## By world

| world | n | invalid_rate | missing_state_rate |
|---|---:|---:|---:|
| tracer_sponge_firm_flat | 120 | 0.458 | 0.000 |

## By action

| world::action | n | invalid_rate | missing_state_rate | top reasons |
|---|---:|---:|---:|---|
| tracer_sponge_firm_flat::sponge_slow_high_clear | 30 | 0.533 | 0.000 | low_base_height:16, folded_joint_posture:FR:2, folded_joint_posture:FL:1 |
| tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_025_013 | 30 | 0.300 | 0.000 | low_base_height:9, folded_joint_posture:FR:2 |
| tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_035_016 | 30 | 0.500 | 0.000 | low_base_height:15, folded_joint_posture:FR:3, metric_suspicious_low_progress_stable:2, folded_joint_posture:FL:1, folded_joint_posture:FL,FR,RL,RR:1, folded_joint_posture:FL,RL:1, large_roll_or_pitch:1 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 30 | 0.500 | 0.000 | low_base_height:15, folded_joint_posture:FL:3, folded_joint_posture:FR:3, folded_joint_posture:FR,RR:1, large_roll_or_pitch:1 |
