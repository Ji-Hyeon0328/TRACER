from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tracer.rewards.manager_terms_v0 as tracer_mdp

from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.flat_env_cfg import UnitreeGo1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import UnitreeGo1RoughEnvCfg



def attach_tracer_slide_reward(rewards, *, weight: float = 0.5) -> None:
    """Attach TRACER slide reward to an Isaac Lab reward config object.

    V2 additionally exposes anti-abandonment terms as explicit reward terms:
    - progress reward encourages moving in the commanded direction.
    - active hold penalty discourages standing still under non-zero commands.
    """
    rewards.tracer_slide_reward = RewTerm(
        func=tracer_mdp.tracer_slide_reward_total,
        weight=weight,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "beta_motion": 1.0 / 3.0,
            "beta_stability": 1.0 / 3.0,
            "beta_energy": 1.0 / 3.0,
            "lambda_energy": 0.5,
            "aux_scale": 1.5,
        },
    )

    rewards.tracer_command_progress_reward = RewTerm(
        func=tracer_mdp.tracer_command_progress_reward,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "active_min_speed": 0.10,
            "active_full_speed": 0.50,
        },
    )

    rewards.tracer_active_hold_penalty = RewTerm(
        func=tracer_mdp.tracer_active_hold_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "active_min_speed": 0.10,
            "active_full_speed": 0.50,
            "hold_speed_threshold": 0.20,
        },
    )


@configclass
class TracerGo1FlatEnvCfg(UnitreeGo1FlatEnvCfg):
    """TRACER wrapper config for the Isaac Lab Go1 flat velocity task.

    This v0 config intentionally keeps the base Isaac Lab Go1 locomotion task mostly intact.
    TRACER reward terms are first validated as standalone torch functions, then wired into
    the manager-based reward system in the next step.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # Keep this small for initial smoke tests.
        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5

        # Disable observation corruption for deterministic TRACER reward/debug probes.
        self.observations.policy.enable_corruption = False

        # Disable random pushes for the first TRACER integration sanity checks.
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Initial reward shaping remains close to Isaac Lab Go1.
        # TRACER slide reward will be added as explicit terms after env creation works.
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.dof_torques_l2.weight = -0.0002
        attach_tracer_slide_reward(self.rewards, weight=0.5)


@configclass
class TracerGo1RoughEnvCfg(UnitreeGo1RoughEnvCfg):
    """TRACER wrapper config for the Isaac Lab Go1 rough velocity task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Keep this small for initial smoke tests.
        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5

        # Disable observation corruption and pushes for deterministic reward/debug probes.
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Keep rough terrain generator from parent.
        # Later we will add slope/friction/deformable-like terrain variants here.

        # Initial reward shaping remains close to Isaac Lab Go1.
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_torques_l2.weight = -0.0002
        attach_tracer_slide_reward(self.rewards, weight=0.5)


@configclass
class TracerGo1FlatForwardEvalEnvCfg(TracerGo1FlatEnvCfg):
    """Forward-only evaluation config for TRACER Go1 flat locomotion."""

    def __post_init__(self):
        super().__post_init__()

        # Keep eval small and deterministic-ish.
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5

        # Force forward command. This separates command-induced backstep
        # from policy/reward-induced backstep.
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Critical for video/eval:
        # the parent velocity task can sample "standing" envs with zero command.
        # Since RecordVideo usually shows env_0, even a small standing probability
        # can make the recorded robot look stuck although the policy is not.
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0

        # Avoid heading command from overriding yaw-rate command if present.
        if hasattr(self.commands.base_velocity, "heading_command"):
            self.commands.base_velocity.heading_command = False
        if hasattr(self.commands.base_velocity.ranges, "heading"):
            self.commands.base_velocity.ranges.heading = (0.0, 0.0)


@configclass
class TracerGo1RoughForwardEvalEnvCfg(TracerGo1RoughEnvCfg):
    """Forward-only evaluation config for TRACER Go1 rough locomotion."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0

        if hasattr(self.commands.base_velocity, "heading_command"):
            self.commands.base_velocity.heading_command = False
        if hasattr(self.commands.base_velocity.ranges, "heading"):
            self.commands.base_velocity.ranges.heading = (0.0, 0.0)

