# ICRA27: TRACER Low-Level Controller Separation

## 1. Scope

This branch isolates, validates, and documents the low-level locomotion controller boundary for TRACER.

The immediate objective is to build a closed low-level control box that:

1. receives interpretable high-level gait meta-commands,
2. maps them into feasible controller references,
3. generates whole-body motor commands through Quadruped-PyMPC, and
4. demonstrates measurable command authority in MuJoCo.

This branch is intentionally separated from the training and evaluation of the full TRACER high-level stack.

---

## 2. Target Architecture

```text
TRACER High-Level Planner
    |
    | MetaGaitCommand theta
    | - target velocity
    | - yaw rate
    | - body height
    | - swing clearance
    | - duty factor
    | - gait frequency or period
    v
TRACER Adapter / Gait Reference Generator
    |
    | - safety projection
    | - range and rate constraints
    | - gait-reference generation
    | - controller-specific mapping
    v
Quadruped-PyMPC
    |
    | - SRBD MPC
    | - contact and foothold planning
    | - whole-body interface
    | - desired joint states
    | - motor torques
    v
MuJoCo Quadruped Simulation
    |
    +---------------------- state feedback
```

The TRACER high-level planner does not directly output joint positions or motor torques.

---

## 3. Controller Boundary

### Input

The initial controller input is a structured `MetaGaitCommand`:

```text
vx_ref
vy_ref
yaw_rate_ref
body_height_ref
swing_clearance_ref
duty_factor_ref
gait_frequency_ref
```

Additional parameters such as step length, stance width, gait type, or controller weights may be introduced only after the initial command-authority tests are complete.

### Output

The low-level controller output is:

```text
motor_torque[12]
```

The following internal values should also be exposed as diagnostics:

```text
desired_joint_position[12]
desired_joint_velocity[12]
desired_joint_acceleration[12]
desired_ground_reaction_force[12]
planned_foothold[4, 3]
contact_sequence
projected_meta_gait_command
```

---

## 4. In Scope

- Reproducible standalone Quadruped-PyMPC execution in MuJoCo
- A fixed-command walking baseline
- A controller-independent `MetaGaitCommand` schema
- Safety projection and rate limiting of gait commands
- Mapping from TRACER meta-actions to PyMPC references
- Independent command-authority tests for:
  - forward velocity
  - body height
  - swing clearance
  - duty factor
  - gait frequency or period
- Closed-loop motor-torque execution
- Runtime logging of commanded, projected, and realized gait quantities
- A ROS 2 wrapper after the standalone MuJoCo loop is stable
- Comparison with the legacy A1-QP-MPC baseline where appropriate

---

## 5. Out of Scope

The following components are not part of the initial low-level validation:

- Objective Selector training
- RAM training
- GMS training
- High-level reinforcement-learning training
- Terrain perception
- Oracle terrain scheduling
- Preference learning or inverse reinforcement learning
- Deformable-terrain claims
- Full Gazebo-to-MuJoCo result equivalence
- Real-robot deployment

These components will be reconnected only after the low-level interface and action authority are validated.

---

## 6. Reused TRACER Components

The first branch scaffold reuses the following existing files:

```text
tracer_core/highlevel/theta_direct_mapper.py
tracer_core/highlevel/theta_safety_projector.py
scripts/runtime/tracer_check_theta_direct_to_ref_v0.py
scripts/runtime/tracer_run_theta_sweep_rollouts_v0.py
```

The current `tracer_core/highlevel` location is retained temporarily to preserve existing imports.

After behavior-preserving tests are added, the mapper and safety projector should be moved to a controller-interface namespace such as:

```text
tracer_core/lowlevel/reference_generation/
```

Legacy body-height candidate configurations may be retained only as characterization seeds and should not be treated as validated PyMPC parameters.

---

## 7. External Baseline

Quadruped-PyMPC is included as a pinned Git submodule:

```text
external_baselines/Quadruped-PyMPC
```

The pinned upstream revision used to initialize this branch is:

```text
dc13d9db3dc9294458d279486fb394aa1c448fc2
```

Upstream changes must be reviewed and introduced through explicit submodule revision updates.

---

## 8. Milestones

### M0 — Reproducible Upstream Baseline

- Install the official Quadruped-PyMPC environment
- Run the original MuJoCo simulation without TRACER modifications
- Record robot model, gait, controller mode, simulation rate, and solver timing
- Save a minimal reproducibility command

### M1 — Fixed TRACER Command

- Introduce a fixed `MetaGaitCommand`
- Route it through the TRACER adapter
- Reproduce stable forward walking
- Verify that the projected command matches the controller input

### M2 — Action-Authority Characterization

Independently sweep:

```text
vx_ref
body_height_ref
swing_clearance_ref
duty_factor_ref
gait_frequency_ref
```

For each parameter, compare:

```text
commanded value
projected value
controller reference
realized robot response
```

A parameter is not considered an active TRACER action unless its effect is repeatable and measurable.

### M3 — Dynamic Meta-Gait Commands

- Apply piecewise-constant gait commands
- Add interpolation and rate limits
- Verify stable transitions between command regions
- Detect rejected, clipped, or ineffective commands

### M4 — ROS 2 Low-Level Node

Create a low-level controller node with a controller-independent interface:

```text
/tracer/meta_gait_command
/tracer/robot_state
/tracer/lowlevel_diagnostics
/tracer/motor_command
```

The ROS 2 wrapper must not contain high-level policy logic.

### M5 — High-Level Reconnection

Reconnect a minimal high-level policy only after M0–M4 pass.

The first high-level comparison should use:

1. fixed meta-gait command,
2. manually scheduled command,
3. learned high-level command.

Objective Selector, RAM, and GMS should remain disabled during the first comparison.

---

## 9. Success Criteria

The initial low-level box is considered closed when:

1. the upstream PyMPC MuJoCo baseline is reproducible;
2. TRACER commands reach the controller through one explicit adapter path;
3. the controller generates stable motor torques;
4. command and state timestamps are logged;
5. at least velocity, swing clearance, and one temporal gait parameter show measurable authority;
6. ineffective parameters are explicitly identified rather than assumed to work;
7. the fixed-command baseline can be rerun from a clean checkout.

---

## 10. Repository Policy

Generated rollouts, solver logs, videos, trained models, and large evaluation reports must not be committed directly to this branch.

The Git history should primarily contain:

- source code,
- compact configuration files,
- interface definitions,
- tests,
- documentation,
- small reproducibility metadata.

Large artifacts should be stored under the NAS experiment directory and referenced by a manifest containing the commit hash, configuration hash, and execution command.