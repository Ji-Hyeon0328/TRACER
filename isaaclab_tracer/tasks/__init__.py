from __future__ import annotations

import gymnasium as gym


GO1_RSL_RL_CFG = "isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents.rsl_rl_ppo_cfg"
GO1_SKRL_AGENTS = "isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents"


gym.register(
    id="Isaac-TRACER-Velocity-Flat-Unitree-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tracer.tasks.go1_velocity_tracer_env_cfg:TracerGo1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{GO1_RSL_RL_CFG}:UnitreeGo1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{GO1_SKRL_AGENTS}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-TRACER-Velocity-Rough-Unitree-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tracer.tasks.go1_velocity_tracer_env_cfg:TracerGo1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{GO1_RSL_RL_CFG}:UnitreeGo1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{GO1_SKRL_AGENTS}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-TRACER-Velocity-Flat-ForwardEval-Unitree-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tracer.tasks.go1_velocity_tracer_env_cfg:TracerGo1FlatForwardEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{GO1_RSL_RL_CFG}:UnitreeGo1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{GO1_SKRL_AGENTS}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-TRACER-Velocity-Rough-ForwardEval-Unitree-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tracer.tasks.go1_velocity_tracer_env_cfg:TracerGo1RoughForwardEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{GO1_RSL_RL_CFG}:UnitreeGo1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{GO1_SKRL_AGENTS}:skrl_rough_ppo_cfg.yaml",
    },
)
