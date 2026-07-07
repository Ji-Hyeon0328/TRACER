# Phase-B Invalid Gait Detection v1

- Root: `artifacts/phase_b_common_action_bank_pilot_v1`
- Episodes: `360`
- Invalid: `221`
- Missing state: `0`

## By world

| world | n | invalid_rate | missing_state_rate |
|---|---:|---:|---:|
| earth | 60 | 0.550 | 0.000 |
| stairs_single | 60 | 0.617 | 0.000 |
| tracer_rough_low | 60 | 0.567 | 0.000 |
| tracer_rough_mid | 60 | 0.517 | 0.000 |
| tracer_slippery_flat | 60 | 1.000 | 0.000 |
| tracer_sponge_firm_flat | 60 | 0.433 | 0.000 |

## By action

| world::action | n | invalid_rate | missing_state_rate | top reasons |
|---|---:|---:|---:|---|
| earth::sponge_slow_high_clear | 10 | 1.000 | 0.000 | metric_suspicious_low_progress_stable:10 |
| earth::sponge_v1d_bias_late_hold_025_013 | 10 | 0.300 | 0.000 | metric_suspicious_low_progress_stable:3 |
| earth::sponge_v1d_stabilized_late_hold_035_016 | 10 | 1.000 | 0.000 | metric_suspicious_low_progress_stable:10 |
| earth::sponge_v8b_reach_bias | 10 | 1.000 | 0.000 | metric_suspicious_low_progress_stable:10 |
| earth::trot_mid | 10 | 0.000 | 0.000 |  |
| earth::trot_solid_fast | 10 | 0.000 | 0.000 |  |
| stairs_single::sponge_slow_high_clear | 10 | 1.000 | 0.000 | metric_suspicious_low_progress_stable:10 |
| stairs_single::sponge_v1d_bias_late_hold_025_013 | 10 | 0.400 | 0.000 | large_roll_or_pitch:3, metric_suspicious_low_progress_stable:3, low_base_height:2, folded_joint_posture:FL,FR:1 |
| stairs_single::sponge_v1d_stabilized_late_hold_035_016 | 10 | 0.900 | 0.000 | metric_suspicious_low_progress_stable:9 |
| stairs_single::sponge_v8b_reach_bias | 10 | 1.000 | 0.000 | metric_suspicious_low_progress_stable:10, folded_joint_posture:FR,RR:1, large_roll_or_pitch:1, low_base_height:1 |
| stairs_single::trot_mid | 10 | 0.100 | 0.000 | folded_joint_posture:FL,FR,RL:1, large_roll_or_pitch:1, low_base_height:1, metric_suspicious_low_progress_stable:1 |
| stairs_single::trot_solid_fast | 10 | 0.300 | 0.000 | large_roll_or_pitch:3, low_base_height:1 |
| tracer_rough_low::sponge_slow_high_clear | 10 | 1.000 | 0.000 | metric_suspicious_low_progress_stable:10 |
| tracer_rough_low::sponge_v1d_bias_late_hold_025_013 | 10 | 0.200 | 0.000 | metric_suspicious_low_progress_stable:2 |
| tracer_rough_low::sponge_v1d_stabilized_late_hold_035_016 | 10 | 1.000 | 0.000 | metric_suspicious_low_progress_stable:10 |
| tracer_rough_low::sponge_v8b_reach_bias | 10 | 1.000 | 0.000 | metric_suspicious_low_progress_stable:10 |
| tracer_rough_low::trot_mid | 10 | 0.000 | 0.000 |  |
| tracer_rough_low::trot_solid_fast | 10 | 0.200 | 0.000 | large_roll_or_pitch:2, low_base_height:2, metric_suspicious_low_progress_stable:2, folded_joint_posture:FL,FR:1 |
| tracer_rough_mid::sponge_slow_high_clear | 10 | 1.000 | 0.000 | metric_suspicious_low_progress_stable:10 |
| tracer_rough_mid::sponge_v1d_bias_late_hold_025_013 | 10 | 0.100 | 0.000 | metric_suspicious_low_progress_stable:1 |
| tracer_rough_mid::sponge_v1d_stabilized_late_hold_035_016 | 10 | 1.000 | 0.000 | metric_suspicious_low_progress_stable:10 |
| tracer_rough_mid::sponge_v8b_reach_bias | 10 | 1.000 | 0.000 | metric_suspicious_low_progress_stable:10 |
| tracer_rough_mid::trot_mid | 10 | 0.000 | 0.000 |  |
| tracer_rough_mid::trot_solid_fast | 10 | 0.000 | 0.000 |  |
| tracer_slippery_flat::sponge_slow_high_clear | 10 | 1.000 | 0.000 | large_roll_or_pitch:10, low_base_height:10, metric_suspicious_low_progress_stable:10, folded_joint_posture:RL:9, folded_joint_posture:FR,RL:1 |
| tracer_slippery_flat::sponge_v1d_bias_late_hold_025_013 | 10 | 1.000 | 0.000 | low_base_height:10, metric_suspicious_low_progress_stable:10, large_roll_or_pitch:8, folded_joint_posture:RL:7, folded_joint_posture:FL,FR,RL,RR:1, folded_joint_posture:FL,RL:1, folded_joint_posture:RL,RR:1, per_leg_motion_collapse:1 |
| tracer_slippery_flat::sponge_v1d_stabilized_late_hold_035_016 | 10 | 1.000 | 0.000 | low_base_height:10, metric_suspicious_low_progress_stable:9, folded_joint_posture:RL:8, large_roll_or_pitch:8, folded_joint_posture:FL,RL:2, per_leg_motion_collapse:1 |
| tracer_slippery_flat::sponge_v8b_reach_bias | 10 | 1.000 | 0.000 | low_base_height:10, metric_suspicious_low_progress_stable:9, large_roll_or_pitch:8, folded_joint_posture:RL:7, folded_joint_posture:FR,RL,RR:2, per_leg_motion_collapse:2, contact_asymmetry:1, folded_joint_posture:RL,RR:1 |
| tracer_slippery_flat::trot_mid | 10 | 1.000 | 0.000 | low_base_height:10, metric_suspicious_low_progress_stable:10, large_roll_or_pitch:9, folded_joint_posture:RL:8, per_leg_motion_collapse:2, contact_asymmetry:1, folded_joint_posture:FL,RL,RR:1, folded_joint_posture:RL,RR:1 |
| tracer_slippery_flat::trot_solid_fast | 10 | 1.000 | 0.000 | large_roll_or_pitch:10, low_base_height:10, metric_suspicious_low_progress_stable:10, folded_joint_posture:RL:8, folded_joint_posture:FL,RL:2, per_leg_motion_collapse:1 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 10 | 0.700 | 0.000 | low_base_height:7 |
| tracer_sponge_firm_flat::sponge_v1d_bias_late_hold_025_013 | 10 | 0.500 | 0.000 | low_base_height:5, folded_joint_posture:FL:2, metric_suspicious_low_progress_stable:1 |
| tracer_sponge_firm_flat::sponge_v1d_stabilized_late_hold_035_016 | 10 | 0.500 | 0.000 | low_base_height:5, folded_joint_posture:FR:1 |
| tracer_sponge_firm_flat::sponge_v8b_reach_bias | 10 | 0.500 | 0.000 | low_base_height:5, folded_joint_posture:FR,RR:1 |
| tracer_sponge_firm_flat::trot_mid | 10 | 0.200 | 0.000 | low_base_height:2, folded_joint_posture:FL:1 |
| tracer_sponge_firm_flat::trot_solid_fast | 10 | 0.200 | 0.000 | low_base_height:2 |
