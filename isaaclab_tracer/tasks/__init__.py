from __future__ import annotations

import gymnasium as gym


# Minimal env-only registration.
# RL agent config entry points will be added after the env construction smoke test passes.
gym.register(
    id="Isaac-TRACER-Velocity-Flat-Unitree-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tracer.tasks.go1_velocity_tracer_env_cfg:TracerGo1FlatEnvCfg",
    },
)

gym.register(
    id="Isaac-TRACER-Velocity-Rough-Unitree-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tracer.tasks.go1_velocity_tracer_env_cfg:TracerGo1RoughEnvCfg",
    },
)
