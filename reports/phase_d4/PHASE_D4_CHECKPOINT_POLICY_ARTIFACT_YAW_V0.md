# TRACER Phase-D4 Checkpoint: Context-Meta Policy Artifact with Yaw Correction v0

## Branch

- branch: `telecommunicating-structure`

## Final policy artifact

- policy JSON: `models/phase_d4/d4_context_meta_policy_rough_clearance_holdvx0025_v0.json`
- runtime selector: `scripts/runtime/tracer_phase_d4_context_meta_selector_node_v0.py`

## Final runtime interface

The D4 selector publishes:

```text
/tracer/mpc_reference
std_msgs/Float64MultiArray
[seq, vx, yaw_rate, body_h, clearance, enable]
Selected context meta policy
flat:      vx=0.2100, body_h=0.320, clearance=0.045
start:     vx=0.2100, body_h=0.320, clearance=0.045
upslope:   vx=0.2100, body_h=0.320, clearance=0.045
rough:     vx=0.2050, body_h=0.320, clearance=0.055
downslope: vx=0.2025, body_h=0.320, clearance=0.045
goal_flat: vx=0.2025, body_h=0.320, clearance=0.045
unknown:   vx=0.2025, body_h=0.320, clearance=0.045
Selected hold policy
hold_vx = 0.025
Selected yaw correction
yaw_gain = -0.025
yaw_sign = -1.0
yaw_k    = 0.025
yaw_max  = 0.20
Key validation results
D4.1 Active hold velocity
selected: hold_vx=0.025
final N=5:
success_rate: 1.000
goal_rate: 1.000
hold_drift_mean: 0.0168 m
D4.2 Context table search
selected: rough_clearance
final N=5:
success_rate: 1.000
goal_rate: 1.000
startup_failed_rate: 0.000
out_lane_rate: 0.000
hold_drift_mean: 0.0414 m
D4.3 Policy artifact runtime loading
policy JSON loaded by runtime selector
N=3:
success_rate: 1.000
goal_rate: 1.000
startup_failed_rate: 0.000
out_lane_rate: 0.000
hold_drift_mean: 0.0450 m
D4.4 Yaw correction search
selected: yaw_gain=-0.025
final N=5:
success_rate: 1.000
goal_rate: 1.000
startup_failed_rate: 0.000
out_lane_rate: 0.000
hold_drift_mean: 0.0398 m
D4.4 Policy JSON with embedded yaw correction
N=3:
success_rate: 1.000
goal_rate: 1.000
startup_failed_rate: 0.000
out_lane_rate: 0.000
hold_drift_mean: 0.0547 m
Interpretation

Phase-D4 closes a terrain-context-conditioned high-level meta-action loop:

terrain context label
  -> rollout-selected policy artifact
  -> theta = [vx, body_h, clearance]
  -> hold velocity
  -> lateral yaw correction
  -> 6-field MPC reference
  -> Gazebo/A1-QP-MPC validation

This is not yet a neural RL policy. It is a rollout-selected contextual meta-policy artifact. It can be used as a teacher, safe prior, or initialization point for the later learned Objective Selector / RAM-conditioned high-level planner.

Known limitations
Current terrain context is oracle/context-label based.
Policy is table-based, not yet learned from perception/RAM.
Yaw correction improves average lateral behavior but does not eliminate lateral drift.
Validated on tracer_mixed_stress_course_v5_lowfric_from_solid; deformable/icy variants still need separate evaluation.
