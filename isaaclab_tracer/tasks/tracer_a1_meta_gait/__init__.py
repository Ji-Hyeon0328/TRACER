from __future__ import annotations

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-TRACER-A1-MetaGait-v0",
    entry_point=(
        "isaaclab_tracer.tasks.tracer_a1_meta_gait."
        "tracer_a1_meta_gait_env_cfg:TracerA1MetaGaitEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tracer.tasks.tracer_a1_meta_gait."
            "tracer_a1_meta_gait_env_cfg:TracerA1MetaGaitEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:TracerA1MetaGaitPPORunnerCfg"
        ),
    },
)
