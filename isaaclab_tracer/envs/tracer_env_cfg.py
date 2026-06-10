from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TracerGoalCfg:
    relative_goal: bool = True
    x_range: Tuple[float, float] = (0.5, 2.0)
    y_range: Tuple[float, float] = (-1.0, 1.0)
    success_tolerance: float = 0.30


@dataclass
class TracerTerrainCfg:
    use_flat: bool = True
    use_rough: bool = True
    use_slope: bool = True
    friction_range: Tuple[float, float] = (0.4, 1.2)
    roughness_range: Tuple[float, float] = (0.0, 0.08)
    slope_range_deg: Tuple[float, float] = (-12.0, 12.0)


@dataclass
class TracerHighLevelCfg:
    history_len: int = 40
    history_feature_dim: int = 11
    context_dim: int = 16
    goal_dim: int = 2
    latent_dim: int = 32
    theta_dim: int = 6
    max_vx: float = 0.12
    max_yaw: float = 0.20


@dataclass
class TracerActionCfg:
    action_type: str = "joint_position_residual"
    action_dim: int = 12
    clip: float = 1.0
    residual_scale: float = 0.25


@dataclass
class TracerRewardCfg:
    goal_progress: float = 1.0
    velocity_tracking: float = 0.5
    yaw_tracking: float = 0.3
    stability: float = 0.5
    energy_penalty: float = -0.01
    fall_penalty: float = -5.0
    success_bonus: float = 5.0


@dataclass
class TracerEnvCfg:
    name: str = "tracer_goal_locomotion_v0"
    num_envs: int = 4096
    episode_length_sec: float = 20.0
    control_dt: float = 0.02
    decimation: int = 4

    goal: TracerGoalCfg = TracerGoalCfg()
    terrain: TracerTerrainCfg = TracerTerrainCfg()
    high_level: TracerHighLevelCfg = TracerHighLevelCfg()
    action: TracerActionCfg = TracerActionCfg()
    reward: TracerRewardCfg = TracerRewardCfg()
