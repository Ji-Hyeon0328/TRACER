# TRACER Isaac Lab integration

This directory is the Isaac Lab-side integration layer for TRACER.

Current purpose:
- Port TRACER reward structure into Isaac Lab.
- Build a beta-conditioned meta-gait RL task.
- Reuse Gazebo rollout reward/preference definitions as offline validation signals.

Current status:
- Gazebo mission logs -> rollout metrics: available.
- Rollout manifest: available.
- TRACER slide reward computation: available.
- Slide reward preference pairs: available.
- Learned slide preference reward model: available.
- Isaac Lab task skeleton: in progress.

Planned structure:
- `tasks/`: Isaac Lab task registration and environment configs.
- `rewards/`: TRACER reward components.
- `assets/`: robot/terrain asset notes or wrappers.
- `configs/`: TRACER-specific Isaac Lab task configs.

Reward target:
R = (beta_motion * R_motion + beta_stability * R_stability + beta_energy * lambda_E * R_energy) / N
    * exp(-c_aux * R_aux)

Notes:
- In Gazebo v0, energy is a proxy.
- In Isaac Lab, energy should be computed from torque, joint velocity, or actuator effort.
- Online RL should not use post-hoc trajectory_cost_v0 as a policy-time reward feature.
