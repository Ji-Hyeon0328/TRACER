from __future__ import annotations

import gymnasium as gym

from isaaclab_tasks.manager_based.locomotion.velocity.config.go1 import agents as go1_agents


gym.register(
    id="Isaac-TRACER-Velocity-Flat-Unitree-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tracer.tasks.go1_velocity_tracer_env_cfg:TracerGo1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{go1_agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{go1_agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-TRACER-Velocity-Rough-Unitree-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tracer.tasks.go1_velocity_tracer_env_cfg:TracerGo1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{go1_agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{go1_agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
