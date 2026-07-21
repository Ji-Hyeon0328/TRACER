# TRACER Phase-J0 Meta-Action Bank v0

This defines a discrete high-level meta-action bank for future contextual-bandit / offline-RL meta-plan learning.

- output config: `configs/phase_j/meta_action_bank_v0.json`
- number of actions: `9`
- active control: `false`
- intended next step: J1 offline/surrogate action scoring

## Safety projection

- vx: `0.025` to `0.22`
- yaw_rate: `-0.2` to `0.2`
- body_h: `0.3` to `0.34`
- clearance: `0.035` to `0.065`
- hold_vx: `0.025`

## Actions

| id | name | intended contexts | vx scale | body_h delta | clearance delta | stability bias | energy bias | hold |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 0 | nominal_cruise | flat, start_flat, goal_flat | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| 1 | fast_motion | flat, start_flat | 1.080 | 0.000 | -0.003 | -0.100 | -0.050 | False |
| 2 | energy_saver | flat, goal_flat | 0.880 | 0.000 | -0.004 | 0.000 | 0.150 | False |
| 3 | rough_high_clearance | rough | 0.920 | 0.000 | 0.010 | 0.150 | -0.050 | False |
| 4 | rough_stability | rough | 0.820 | -0.005 | 0.012 | 0.250 | -0.100 | False |
| 5 | upslope_push | upslope | 0.960 | 0.000 | 0.004 | 0.120 | -0.050 | False |
| 6 | downslope_stable | downslope | 0.780 | -0.006 | 0.004 | 0.300 | -0.100 | False |
| 7 | lateral_recovery_soft | rough, downslope, unknown | 0.650 | -0.008 | 0.008 | 0.350 | -0.150 | False |
| 8 | goal_hold | goal_flat | 0.000 | 0.000 | 0.000 | 0.250 | 0.100 | True |

## Safe interpretation

- J0 only defines candidate high-level meta-actions.
- J0 does not modify active control.
- The existing D7 Objective Selector should remain the active beta source.
- I5 safe RAM may be used as a shadow/auxiliary feature, but not as active RAM yet.
- J1 should score these actions offline or in shadow before any active rollout.
