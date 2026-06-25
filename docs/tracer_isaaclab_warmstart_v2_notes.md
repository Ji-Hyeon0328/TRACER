# TRACER Isaac Lab Warm-start V2 Notes

## Summary

The from-scratch TRACER reward experiments revealed a pose-hold / task-abandonment local optimum.  
V2 introduced explicit command-progress and active-hold penalty terms, but from-scratch PPO still did not reliably discover a clean locomotion gait in deterministic play.

Warm-starting V2 from the baseline Go1 flat locomotion policy resolved this issue.

## Warm-start V2 result

Experiment:

- Task: Isaac-TRACER-Velocity-Flat-Unitree-Go1-v0
- Init checkpoint: baseline_go1_flat_v0 / model_499.pt
- Reward: TRACER V2 with explicit anti-abandonment terms
- Run: tracer_go1_flat_tracer_reward_v2_warmstart / baseline_init_plus500

Final observed metrics:

- Mean reward: approximately 48–50
- Mean episode length: approximately 955–991
- tracer_slide_reward: approximately 0.23–0.28
- tracer_command_progress_reward: approximately 0.76–0.93
- tracer_active_hold_penalty: approximately -0.005 to -0.010
- error_vel_xy: approximately 0.22–0.25
- time_out: approximately 0.965–0.973
- base_contact: approximately 0.027–0.035

## Interpretation

This suggests that TRACER reward shaping should not be used as a from-scratch gait discovery objective.  
Instead, TRACER should be used to modulate and fine-tune an existing locomotion prior.

This matches the intended TRACER architecture:

- Isaac Lab: high-level planner, objective selector, context/mismatch-conditioned reward-weight adaptation
- Gazebo/controller stack: low-level gait generation, controller reference tracking, MPC/WBC/impedance-residual validation

## Caveat

The resulting Isaac Lab motion should not yet be treated as the final desired gait.  
The motion validates the reward/planner pipeline, but final gait quality should be handled by the Gazebo controller stack.
