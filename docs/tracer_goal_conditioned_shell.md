# TRACER Goal-Conditioned Adaptive Locomotion Shell

## Current milestone

This document summarizes the current Gazebo prototype of TRACER using the public A1-QP-MPC low-level controller.

The current system closes the following loop:

```text
Goal / relative waypoint
    ↓
GoalTracker
    ↓ raw desired vx/yaw
Robust Adaptation Module
    ↓ rho, sigma
Objective Selector
    ↓ beta_v, beta_s, beta_e, mode
Beta-conditioned modulation
    ↓ safe vx/yaw command
Gazebo Joy Adapter
    ↓ /joy
A1-QP-MPC low-level controller
    ↓
Gazebo A1 robot
## Modules
# GoalTracker

The GoalTracker converts a goal into raw velocity and yaw commands.

Inputs:

current base pose
goal position
current yaw

Outputs:

raw_vx_axis
raw_yaw_axis
distance to goal
heading error

The command law is:
- desired_heading = atan2(goal_y - y, goal_x - x)
- heading_error = wrap_to_pi(desired_heading - yaw)
- raw_yaw_axis = clamp(k_yaw * heading_error)
- raw_vx_axis = clamp(k_v * distance) * yaw_factor
The yaw factor reduces forward motion when the heading error is large.

# Robust Adaptation Module

The current RAM is a proxy module, not learned yet.

Inputs:

- state history proxy
- last commanded velocity

Outputs:

- rho_v_inst
- rho_v_mean
- sigma_v
- v_cmd
- v_meas

Definitions:
rho_v_inst = |v_cmd - v_meas| / max(|v_cmd|, eps)
rho_v_mean = moving average of rho_v_inst
sigma_v = moving standard deviation of rho_v_inst

Current interpretation:
rho  ≈ recent command-tracking mismatch
sigma ≈ variability/uncertainty of recent mismatch

In the future, this proxy RAM can be replaced by a learned RAM that outputs a latent mismatch distribution.

#Objective Selector

The current ObjectiveSelector is rule-based.

Inputs:

- posture stability
- rho_v_mean
- sigma_v

Outputs:

- mode
- beta_v
- beta_s
- beta_e

Current modes:

- nominal
- cautious_mismatch
- conservative_mismatch
- conservative_posture
- recovery

Interpretation:
beta_v: progress / velocity preference
beta_s: stability preference
beta_e: energy / smoothness preference

Beta-conditioned modulation

The GoalTracker proposes a raw velocity command. TRACER modulates it using beta:
nominal = raw_vx_axis
conservative = min(conservative_vx_axis, nominal)
energy = 0.5 * (nominal + conservative)

vx_axis =
    beta_v * nominal
  + beta_s * conservative
  + beta_e * energy
  
Yaw is currently passed from GoalTracker directly, with clipping. Future versions may also beta-modulate yaw.

# Calibration results

Flat-world calibration showed:
vx_axis = 0.12, yaw_axis = 0.0
→ speed ≈ 0.057–0.064 m/s
→ yaw drift small

vx_axis = 0.0, yaw_axis = 0.20
→ yaw rate ≈ +0.15 rad/s

vx_axis = 0.12, yaw_axis = 0.20
→ forward motion and yaw rotation work together

# Successful goal-conditioned runs

Representative successful logs:

Forward goal:
relative_goal_x = 1.0
relative_goal_y = 0.0
reached = True
emergency_stop = 0

Lateral-offset goal:
relative_goal_x = 1.0
relative_goal_y = 0.5
reached = True
emergency_stop = 0

Harder lateral goal:
relative_goal_x = 0.5
relative_goal_y = 1.0
reached = True
emergency_stop = 0

In these runs, TRACER modulation was active:
avg raw_vx_axis > avg vx_axis

This means GoalTracker generated a command, and TRACER reduced it according to mismatch/stability conditions.

# Known issue fixed

The low-level controller uses button[0] as a toggle, not as an explicit walk/stand command.

A bug occurred when shutdown and normal exit both toggled the mode, causing walking mode to be re-enabled. This was fixed by tracking toggle ownership:

walking_mode_owned = True only if this node toggled walking on
toggle off only if walking_mode_owned is True
shutdown hook avoids double-toggle

## Current limitations
1. RAM is proxy-based, not learned.
2. ObjectiveSelector is rule-based, not learned.
3. Yaw command is not beta-modulated yet.
4. GoalTracker uses a simple proportional law.
5. The low-level controller receives commands through /joy, which is convenient but indirect.
6. Performance depends on the internal latch state of A1-QP-MPC walking mode.
7. Gazebo contact and stance behavior can still produce small drift.

## Next steps

Recommended next steps:
M14: Add goal-tracer summary filtering and reporting table.
M15: Add yaw beta-modulation for conservative modes.
M16: Replace proxy RAM with learned/history encoder.
M17: Replace rule-based ObjectiveSelector with learned beta selector.
M18: Evaluate on multiple terrain worlds.

---

## M17-M20: Learned RAM baseline

After validating the proxy RAM version of the goal-conditioned TRACER shell, we added a learned RAM baseline.

### Dataset construction

GoalTracer logs are converted into a sequence dataset:

```text
X: [N, H, F]
Y: [N, 2]
where: 
H = 40 history steps
F = 11 features
Y = [rho_v_mean, sigma_v]

The current feature vector includes:
raw_vx_axis, vx_axis,
raw_yaw_axis, yaw_axis,
goal_dist,
vx_mod_error, yaw_mod_error,
beta_v, beta_s, beta_e,
mode_id

This dataset currently imitates the proxy RAM behavior. It is not yet a privileged teacher-student RAM trained from terrain/contact labels.

## Learned RAM model

The first learned RAM baseline is a numpy ridge-regression model:
history window
→ flattened history vector
→ ridge regression
→ rho_hat, sigma_hat

This is not the final neural RAM architecture, but it verifies that RAM can be represented as a learned estimator.

## Online integration

The learned RAM is integrated into GoalTracer as an optional module:
_use_learned_ram:=true
_learned_ram_model:=<model_path>

The online estimator uses a one-step delayed structure:
previous finalized command history
→ learned RAM predicts rho/sigma
→ ObjectiveSelector chooses beta/mode
→ TRACER computes vx/yaw
→ finalized command is appended to RAM history

This avoids circular dependency because RAM features include beta, mode, and finalized commands.

## Verified behavior

The learned RAM version was tested on three relative goal tasks:
Forward 1.0 m
Forward 1.0 m + lateral 0.5 m
Forward 0.5 m + lateral 1.0 m

All tasks were reached with no emergency stop.

The logs include ram_source, which records whether each timestep used:
proxy_warmup
learned
reached

This confirms that after the initial warmup window, the online ObjectiveSelector uses learned RAM predictions.

## Current interpretation

The learned RAM baseline should not yet be claimed as a full sim-to-real adaptation module. The current result shows:
A learned RAM estimator can replace the proxy RAM in the online TRACER loop
while preserving goal-reaching behavior in Gazebo.

The next step is to train a stronger learned RAM in Isaac Lab using richer terrain/contact histories and privileged labels.
