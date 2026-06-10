# TRACER Closed-loop Integration with A1-QP-MPC

## Current milestone

This integration closes the first TRACER loop in Gazebo:

```text
/body_pose_ground_truth
    -> RuleBasedHighLevelPlanner
    -> ThetaToRefMapper
    -> GazeboJoyAdapter
    -> /joy
    -> A1-QP-MPC low-level controller
    -> Gazebo A1
Confirmed:

TRACER bridge receives odometry from /body_pose_ground_truth.
TRACER bridge toggles walking mode once through /joy buttons[0].
TRACER bridge publishes forward command through /joy axes[5].
A1-QP-MPC receives the command and generates walking motion.
First successful closed-loop command used:
_nominal_vx_axis:=0.03
_conservative_vx_axis:=0.01
_vx_max:=0.05
The first high-level planner is rule-based.

stable base pose -> nominal forward command
moderate roll/pitch -> conservative forward command
low body height or large roll/pitch -> emergency stop

This module is a placeholder for the future Objective Selector, RAM, GMS, and Θ-Decoder.
## Next target
Replace the rule-based planner with TRACER modules gradually:

Add logging of state, theta, and RefCommand.
Add command sweep experiments.
Add safety gate.
Replace rule-based theta generation with Objective Selector / RAM / Θ-Decoder.
