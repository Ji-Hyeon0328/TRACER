from dataclasses import dataclass, field
from typing import Tuple


try:
    from isaaclab.utils import configclass
except Exception:
    def configclass(cls):
        return dataclass(cls)


@configclass
class TracerGoalCommandCfg:
    relative_goal: bool = True
    x_range: Tuple[float, float] = (0.5, 2.0)
    y_range: Tuple[float, float] = (-1.0, 1.0)
    success_tolerance: float = 0.30


@configclass
class TracerHighLevelModuleCfg:
    history_len: int = 40
    history_feature_dim: int = 11
    context_dim: int = 16
    goal_dim: int = 2
    latent_dim: int = 32
    theta_dim: int = 6
    max_vx: float = 0.12
    max_yaw: float = 0.20


@configclass
class TracerLowLevelActionCfg:
    action_type: str = "joint_position_residual"
    action_dim: int = 12
    residual_scale: float = 0.25
    clip: float = 1.0


@configclass
class TracerTerrainRandomizationCfg:
    use_flat: bool = True
    use_rough: bool = True
    use_slope: bool = True
    friction_range: Tuple[float, float] = (0.4, 1.2)
    roughness_range: Tuple[float, float] = (0.0, 0.08)
    slope_range_deg: Tuple[float, float] = (-12.0, 12.0)


@configclass
class TracerIsaacEnvCfg:
    """
    Placeholder Isaac Lab-facing config.

    This is intentionally independent from a specific Isaac Lab Env base class.
    In M28, this will be connected to the available Isaac Lab API
    after checking whether DirectRLEnv or ManagerBasedRLEnv is the better fit.
    """

    name: str = "tracer_goal_locomotion_isaac_v0"
    num_envs: int = 4096
    episode_length_sec: float = 20.0
    control_dt: float = 0.02
    decimation: int = 4

    robot_name: str = "unitree_go1_or_a1"
    num_dofs: int = 12
    num_feet: int = 4

    goal: TracerGoalCommandCfg = field(default_factory=TracerGoalCommandCfg)
    high_level: TracerHighLevelModuleCfg = field(default_factory=TracerHighLevelModuleCfg)
    action: TracerLowLevelActionCfg = field(default_factory=TracerLowLevelActionCfg)
    terrain: TracerTerrainRandomizationCfg = field(default_factory=TracerTerrainRandomizationCfg)
