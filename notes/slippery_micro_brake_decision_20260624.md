# TRACER slippery micro-brake decision — 2026-06-24

## Context

We evaluated whether the slippery downhill case can be handled by a high-level fallback primitive using the current command interface:

```text
[vx, yaw_rate, body_height, swing_clearance, enable]
The target condition was tracer_slippery_mid_slope_5deg, interpreted as the case where the downhill direction is behind the robot body when yaw=0. This was used to test a backward/micro-brake style fallback.

Candidates tested
Active-hold variants
True backstep / micro-backstep variants
Micro-brake vx=-0.005, body_height=0.305, swing_clearance=0.055
Physics-time staged vx ramp:
body-height ramp first
vx held at 0.0 initially
vx ramped from 0.0 to -0.005 during the physics pulse
No-vx decomposition:
same body_height and swing_clearance
vx held at 0.0 for the whole physics pulse
Key result

The late-vx staged test confirmed that the vx ramp was applied during the physics pulse, but the result was still only 1/3 valid and 2/3 fallen.

The no-vx decomposition also produced only 1/3 valid and 2/3 fallen. Therefore, the failure is not caused only by the micro-brake velocity. The current high-level command interface is not sufficient to produce a robust fallback for this slippery slope case.

Decision

This branch is closed as:
no_valid_high_level_velocity_primitive
The terrain should remain:
avoid_required
until a lower-level recovery mechanism, stance-control primitive, friction-aware WBC/MPC modification, or explicit reorientation/avoidance planner is available.

Research implication

This is still useful for TRACER. It gives evidence that RAM/gate should not merely choose a more conservative high-level velocity command in this terrain. Instead, the policy should escalate to avoid/recovery because the current primitive set is insufficient.
