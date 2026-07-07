# Phase-B Invalid Gait Calibration v1b

- Input: `data/phase_b_invalid_gait_labels_v1/common_action_bank_pilot_v1_invalid_gait_labels.jsonl`
- Episodes: `360`
- Hard invalid: `76`
- Warning: `214`

## By world

| world | n | hard_invalid_rate | warning_rate | missing_state_rate |
|---|---:|---:|---:|---:|
| earth | 60 | 0.000 | 0.550 | 0.000 |
| stairs_single | 60 | 0.133 | 0.550 | 0.000 |
| tracer_rough_low | 60 | 0.033 | 0.567 | 0.000 |
| tracer_rough_mid | 60 | 0.000 | 0.517 | 0.000 |
| tracer_slippery_flat | 60 | 1.000 | 0.967 | 0.000 |
| tracer_sponge_firm_flat | 60 | 0.100 | 0.417 | 0.000 |

## By action

| world::action | n | hard_invalid_rate | warning_rate | hard reasons | warning reasons |
|---|---:|---:|---:|---|---|
| earth::sponge_slow_high_clear | 10 | 0.000 | 1.000 |  | metric_suspicious_low_progress_stable:10 |
| earth::sponge_v1d_bias_late_hold_025_013 | 10 | 0.000 | 0.300 |  | metric_suspicious_low_progress_stable:3 |
| earth::sponge_v1d_stabilized_late_hold_035_016 | 10 | 0.000 | 1.000 |  | metric_suspicious_low_progress_stable:10 |
| earth::sponge_v8b_reach_bias | 10 | 0.000 | 1.000 |  | metric_suspicious_low_progress_stable:10 |
| earth::trot_mid | 10 | 0.000 | 0.000 |  |  |
| earth::trot_solid_fast | 10 | 0.000 | 0.000 |  |  |
| stairs_single::sponge_slow_high_clear | 10 | 0.000 | 1.000 |  | metric_suspicious_low_progress_stable:10 |
| stairs_single::sponge_v1d_bias_late_hold_025_013 | 10 | 0.300 | 0.300 | large_roll_or_pitch:3, severe_low_base_height:2, folded_joint_posture:FL,FR:1 | metric_suspicious_low_progress_stable:3 |
| stairs_single::sponge_v1d_stabilized_late_hold_035_016 | 10 | 0.000 | 0.900 |  | metric_suspicious_low_progress_stable:9 |
| stairs_single::sponge_v8b_reach_bias | 10 | 0.100 | 1.000 | folded_joint_posture:FR,RR:1, large_roll_or_pitch:1, severe_low_base_height:1 | metric_suspicious_low_progress_stable:10 |
| stairs_single::trot_mid | 10 | 0.100 | 0.100 | folded_joint_posture:FL,FR,RL:1, large_roll_or_pitch:1, severe_low_base_height:1 | metric_suspicious_low_progress_stable:1 |
| stairs_single::trot_solid_fast | 10 | 0.300 | 0.000 | large_roll_or_pitch:3, severe_low_base_height:1 |  |
| tracer_rough_low::sponge_slow_high_clear | 10 | 0.000 | 1.000 |  | metric_suspicious_low_progress_stable:10 |
| tracer_rough_low::sponge_v1d_bias_late_hold_025_013 | 10 | 0.000 | 0.200 |  | metric_suspicious_low_progress_stable:2 |
| tracer_rough_low::sponge_v1d_stabilized_late_hold_035_016 | 10 | 0.000 | 1.000 |  | metric_suspicious_low_progress_stable:10 |
| tracer_rough_low::sponge_v8b_reach_bias | 10 | 0.000 | 1.000 |  | metric_suspicious_low_progress_stable:10 |
| tracer_rough_low::trot_mid | 10 | 0.000 | 0.000 |  |  |
| tracer_rough_low::trot_solid_fast | 10 | 0.200 | 0.200 | large_roll_or_pitch:2, severe_low_base_height:2, folded_joint_posture:FL,FR:1 | metric_suspicious_low_progress_stable:2 |
| tracer_rough_mid::sponge_slow_high_clear | 10 | 0.000 | 1.000 |  | metric_suspicious_low_progress_stable:10 |
| tracer_rough_mid::sponge_v1d_bias_late_hold_025_013 | 10 | 0.000 | 0.100 |  | metric_suspicious_low_progress_stable:1 |
| tracer_rough_mid::sponge_v1d_stabilized_late_hold_035_016 | 10 | 0.000 | 1.000 |  | metric_suspicious_low_progress_stable:10 |
| tracer_rough_mid::sponge_v8b_reach_bias | 10 | 0.000 | 1.000 |  | metric_suspicious_low_progress_stable:10 |
| tracer_rough_mid::trot_mid | 10 | 0.000 | 0.000 |  |  |
| tracer_rough_mid::trot_solid_fast | 10 | 0.000 | 0.000 |  |  |
| tracer_slippery_flat::sponge_slow_high_clear | 10 | 1.000 | 1.000 | large_roll_or_pitch:10, severe_low_base_height:10, folded_joint_posture:RL:9, folded_joint_posture:FR,RL:1 | metric_suspicious_low_progress_stable:10 |
| tracer_slippery_flat::sponge_v1d_bias_late_hold_025_013 | 10 | 1.000 | 1.000 | severe_low_base_height:10, large_roll_or_pitch:8, folded_joint_posture:RL:7, folded_joint_posture:FL,FR,RL,RR:1, folded_joint_posture:FL,RL:1, folded_joint_posture:RL,RR:1, per_leg_motion_collapse:1 | metric_suspicious_low_progress_stable:10 |
| tracer_slippery_flat::sponge_v1d_stabilized_late_hold_035_016 | 10 | 1.000 | 0.900 | severe_low_base_height:10, folded_joint_posture:RL:8, large_roll_or_pitch:8, folded_joint_posture:FL,RL:2, per_leg_motion_collapse:1 | metric_suspicious_low_progress_stable:9 |
| tracer_slippery_flat::sponge_v8b_reach_bias | 10 | 1.000 | 0.900 | severe_low_base_height:10, large_roll_or_pitch:8, folded_joint_posture:RL:7, folded_joint_posture:FR,RL,RR:2, per_leg_motion_collapse:2, contact_asymmetry:1, folded_joint_posture:RL,RR:1 | metric_suspicious_low_progress_stable:9 |
| tracer_slippery_flat::trot_mid | 10 | 1.000 | 1.000 | severe_low_base_height:10, large_roll_or_pitch:9, folded_joint_posture:RL:8, per_leg_motion_collapse:2, contact_asymmetry:1, folded_joint_posture:FL,RL,RR:1, folded_joint_posture:RL,RR:1 | metric_suspicious_low_progress_stable:10 |
| tracer_slippery_flat::trot_solid_fast | 10 | 1.000 | 1.000 | large_roll_or_pitch:10, severe_low_base_height:10, folded_joint_posture:RL:8, folded_joint_posture:FL,RL:2, per_leg_motion_collapse:1 | metric_suspicious_low_progress_stable:10 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 10 | 0.100 | 0.600 | severe_low_base_height:1 | soft_low_base_height:6 |
| tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_025_013 | 10 | 0.200 | 0.500 | folded_joint_posture:FL:2, severe_low_base_height:1 | soft_low_base_height:4, metric_suspicious_low_progress_stable:1 |
| tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_035_016 | 10 | 0.100 | 0.500 | folded_joint_posture:FR:1 | soft_low_base_height:5 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 10 | 0.100 | 0.500 | folded_joint_posture:FR,RR:1 | soft_low_base_height:5 |
| tracer_sponge_firm_flat::trot_mid | 10 | 0.100 | 0.200 | folded_joint_posture:FL:1 | soft_low_base_height:2 |
| tracer_sponge_firm_flat::trot_solid_fast | 10 | 0.000 | 0.200 |  | soft_low_base_height:2 |
