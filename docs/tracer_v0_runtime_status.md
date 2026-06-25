# TRACER V0 Runtime Status

## Current runtime skeleton

The current TRACER V0 runtime skeleton connects the following modules:

- RAM latent input path: rho, sigma can be injected into the policy input.
- Objective Selector beta path: beta can be injected into the policy input.
- Gait Mode Selector: rule-based GMS is available.
- Meta-gait policy interface:
  - rule_based
  - learned MLP warm-start
  - UDP client/server policy interface
- Decoder / Mapper:
  - maps meta-gait output to the current 6D MPC reference:
    [counter, vx, yaw_rate, body_height, swing_clearance, enable]
- Fusion policy publisher:
  - publishes /tracer/mpc_reference
  - supports GMS/meta-gait path
  - supports primitive bypass for hand-authored backstep / active-hold primitives.

## Important clarification

The current learned meta-gait policy is not yet a true RL policy.
It is a supervised warm-start model trained from rule-based / hand-authored references.

The actual remaining learning target is:

state/input:
- terrain context c_t
- RAM latent rho
- RAM uncertainty sigma
- Objective weights beta
- robot state
- local/relative goal

action:
- meta-gait parameter theta
- examples:
  - vx
  - yaw_rate
  - body_height
  - swing_clearance
  - gait_period
  - duty_factor
  - step_length
  - stance_width
  - impedance_scale
  - residual_gain_scale
  - risk_scale

reward:
- beta_v * motion reward
- beta_s * stability reward
- beta_e * energy reward
- fall/slip/contact/recovery penalties

## Slippery slope primitive finding

For tracer_slippery_mid_slope_5deg:

- forward walking is unsafe.
- negative-vx micro/true backstep is not deployable with the current low-level controller.
- vx=0 active-hold is safer.
- best active-hold fallback found so far:
  - vx = 0.0
  - yaw_rate = 0.0
  - body_height = 0.310
  - swing_clearance = 0.065
  - enable = 1.0
  - fall_like = False
  - dx ≈ -0.334
  - dy ≈ 0.016
  - min_z ≈ 0.229

## Remaining major tracks

### Objective Selector / beta learning
- simulation rollout dataset
- trajectory metric extraction
- preference / ranking construction
- IRL or preference learning
- beta predictor training
- beta-conditioned RL validation

### Meta-gait RL policy
- actual RL environment setup
- action theta finalization
- reward design
- PPO/SAC style training
- beta/rho/sigma/c_t-conditioned validation

### Context Encoder
- camera/depth/elevation encoder
- terrain context c_t
- visual context and RAM mismatch contrastive use

### Low-level Controller
- current A1 QP-MPC structure analysis
- TRACER MPC/WBC design
- theta-to-ref expansion
- adaptive residual impedance controller
- RAM-conditioned low-level adaptation
- slip/recovery primitive
- CBF-QP or safety recovery layer
