# Isaac Lab Migration Plan for TRACER

## Goal

The goal of the Isaac Lab migration is to move TRACER from a Gazebo-validated shell into a trainable locomotion framework.

The Gazebo prototype validated:

```text
GoalTracker
→ RAM / learned RAM
→ ObjectiveSelector
→ beta-conditioned vx/yaw modulation
→ public A1-QP-MPC backend

The Isaac Lab version should support:
terrain/context observation
→ learned RAM
→ learned ObjectiveSelector
→ beta or theta-conditioned locomotion reference
→ RL policy / low-level controller

## Current Gazebo status

Completed:
1. Goal-conditioned command generation
2. Proxy RAM from command tracking mismatch
3. Learned RAM ridge baseline
4. Rule-based ObjectiveSelector
5. beta-conditioned vx command modulation
6. mode-conditioned yaw modulation
7. config-driven experiment runner
8. proxy RAM vs learned RAM comparison table

Limitations:
1. No terrain-aware encoder
2. RAM is trained only to imitate proxy RAM
3. ObjectiveSelector is rule-based
4. No explicit theta/meta-plan
5. No beta-conditioned MPC/WBC weights
6. No adaptive residual controller
7. No CBF/recovery controller

## Isaac Lab target architecture
Observation
  proprioception
  command history
  base velocity
  base orientation
  joint position / velocity
  contact state
  terrain height scan or elevation patch
  previous action
        ↓
Context Encoder
        ↓
RAM
  rho, sigma, latent z_r
        ↓
ObjectiveSelector
  beta_v, beta_s, beta_e
        ↓
ThetaDecoder
  gait/reference parameters
        ↓
Theta-to-Ref Mapper
  desired velocity
  body height
  gait timing
  foot placement bias
  impedance/residual gains
        ↓
Low-level policy or controller

## M22 interface decision

For the first Isaac Lab version, keep the action space simple.

## Option A: High-level command modulation

TRACER outputs:
vx_cmd
yaw_cmd
beta_v, beta_s, beta_e

The low-level policy receives these as command inputs.

This is closest to the current Gazebo prototype.

## Option B: Theta-based reference

TRACER outputs:
theta = [
  vx_scale,
  yaw_scale,
  body_height_offset,
  gait_frequency,
  duty_factor,
  stance_width_bias,
  foot_clearance,
  impedance_scale
]

Theta-to-Ref Mapper converts theta into locomotion references.

This is closer to the intended TRACER design.

## Recommended path

Start with Option A, then extend to Option B.

M22/M23:
  beta-conditioned command modulation in Isaac Lab

M24/M25:
  introduce theta and Theta-to-Ref Mapper
  
## Isaac Lab training stages
### Stage 1: Interface reproduction

Reproduce Gazebo behavior in Isaac Lab.

Inputs:
goal direction
distance to goal
base velocity
base orientation
joint state
previous command

Outputs:
vx_cmd
yaw_cmd
beta
mode
rho/sigma

Goal:
match Gazebo-style closed-loop behavior

## Stage 2: Learned RAM

Train RAM using simulation randomization.

Privileged teacher labels may include:
friction
slope
terrain roughness
external push
tracking error
slip estimate
contact inconsistency

Student RAM input:
history of proprioception
history of commands
history of tracking error
contact history

Output:
rho
sigma
latent z_r

## Stage 3: Learned ObjectiveSelector

Replace rule-based ObjectiveSelector.

Input:
terrain context
rho
sigma
goal command
robot state

Output:
beta_v
beta_s
beta_e

Possible training targets:
teacher-selected beta
preference-based beta
reward-conditioned beta
IRL-style beta

## Stage 4: ThetaDecoder

Introduce explicit meta-plan.
beta, rho, sigma, context
→ theta

Theta may include:
gait frequency
duty factor
body height
foot clearance
step length scale
yaw scale
impedance scale

## Stage 5: Controller-level integration

Move from command-level beta to controller-level beta.
beta_v → velocity tracking weight
beta_s → stability/contact margin weight
beta_e → torque/energy/smoothness weight

This is where MPC/WBC integration becomes more aligned with the original TRACER idea.

## Minimum Isaac Lab experiment set

Initial tasks:
1. Flat forward goal
2. Flat lateral-offset goal
3. Rough terrain forward goal
4. Low-friction flat goal
5. Slope up/down goal
6. Random terrain goal

Metrics:

success rate
distance reduction
tracking error
fall rate
energy proxy
slip estimate
rho/sigma distribution
beta distribution
mode distribution

## Key principle

Risk/Time Estimator is not part of the current TRACER core.

It will be handled later as a separate side module.

Current priority:
Finish TRACER core first.
Then revisit path/map-based Risk-Time Estimator.
