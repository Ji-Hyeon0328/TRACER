# TRACER Isaac Lab Training Status V0

## Date

2026-06-24 / 2026-06-25

## Branch

telecommunicating-structure

## Completed

- Registered TRACER Go1 Flat task:
  - `Isaac-TRACER-Velocity-Flat-Unitree-Go1-v0`
- Registered TRACER Go1 Rough task:
  - `Isaac-TRACER-Velocity-Rough-Unitree-Go1-v0`
- Attached `tracer_slide_reward` as an Isaac Lab reward manager term.
- Verified Flat/Rough reset-step smoke tests.
- Added RSL-RL training wrapper:
  - `scripts/runtime/tracer_isaaclab_rslrl_train.py`
- Verified RSL-RL PPO training loop for both Flat and Rough.

## Flat 50-iteration sanity

Command:

```bash
python scripts/runtime/tracer_isaaclab_rslrl_train.py \
  --task Isaac-TRACER-Velocity-Flat-Unitree-Go1-v0 \
  --num_envs 256 \
  --max_iterations 50 \
  --headless \
  --device cuda:0 \
  --experiment_name tracer_go1_flat_tracer_reward_v0 \
  --run_name iter50_env256
Final observed metrics:

Total steps: 307200
Training time: 14.78 seconds
Mean reward: -0.05
Episode_Reward/tracer_slide_reward: 0.2149
Metrics/base_velocity/error_vel_xy: 1.3670
Metrics/base_velocity/error_vel_yaw: 1.0784
Episode_Termination/time_out: 0.9414
Episode_Termination/base_contact: 0.0586
Rough 50-iteration sanity

Command:
python scripts/runtime/tracer_isaaclab_rslrl_train.py \
  --task Isaac-TRACER-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 256 \
  --max_iterations 50 \
  --headless \
  --device cuda:0 \
  --experiment_name tracer_go1_rough_tracer_reward_v0 \
  --run_name iter50_env256
Final observed metrics:

Total steps: 307200
Training time: 27.56 seconds
Mean reward: -0.76
Episode_Reward/tracer_slide_reward: 0.1247
Curriculum/terrain_levels: 2.2598
Metrics/base_velocity/error_vel_xy: 1.6283
Metrics/base_velocity/error_vel_yaw: 1.5811
Episode_Termination/time_out: 0.7617
Episode_Termination/base_contact: 0.2383
Interpretation

The TRACER Isaac Lab V0 integration is now functional.

The custom TRACER reward is active in both Flat and Rough Go1 training tasks, appears in the RSL-RL reward log, and remains finite during 50-iteration PPO sanity runs.

Flat training is stable enough for a longer run. Rough training is also functional, but the high base-contact termination rate suggests that the rough task still needs reward balancing, curriculum tuning, or stabilization improvements before longer training is meaningful.

Next steps
Run a longer Flat training sanity, e.g. 500 iterations.
Compare baseline Go1 Flat vs TRACER Go1 Flat under the same iteration budget.
Tune tracer_slide_reward weight and component scales.
Add explicit actor/critic observation group config to remove RSL-RL warnings.
Extend the TRACER reward from fixed beta to beta-conditioned reward.
Later connect RAM/context inputs into the policy observation.
