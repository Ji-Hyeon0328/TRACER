import math

import torch

from isaaclab_tracer.envs.tracer_task_spec import TracerTaskSpec
from isaaclab_tracer.envs.tracer_highlevel_loop import TracerHighLevelLoop
from isaaclab_tracer.envs.tracer_step_adapter import TracerStepAdapter
from isaaclab_tracer.utils.paths import tracer_config_path


def quat_wxyz_to_rpy(quat):
    """
    Convert quaternion [w, x, y, z] to roll, pitch, yaw.
    Returns [B, 1] tensors.
    """
    w = quat[:, 0:1]
    x = quat[:, 1:2]
    y = quat[:, 2:3]
    z = quat[:, 3:4]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def make_tracer_a1_adapter_env_class():
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.assets import Articulation
    from isaaclab.utils import configclass
    import isaaclab.sim as sim_utils
    from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

    @configclass
    class TracerA1AdapterEnvCfg(DirectRLEnvCfg):
        episode_length_s = 20.0
        decimation = 4

        action_space = 12
        observation_space = 58
        state_space = 0

        sim: SimulationCfg = SimulationCfg(
            dt=0.005,
            render_interval=decimation,
        )

        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=4,
            env_spacing=2.5,
        )

        robot = UNITREE_A1_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
        )

        tracer_goal_x = 1.0
        tracer_goal_y = 0.5
        residual_scale = 0.10

    class TracerA1AdapterEnv(DirectRLEnv):
        cfg: TracerA1AdapterEnvCfg

        def __init__(self, cfg: TracerA1AdapterEnvCfg, render_mode=None, **kwargs):
            self.robot = None

            self._actions = None
            self._previous_action = None
            self._policy_obs = None
            self._reward = None
            self._terminated = None
            self._truncated = None
            self._joint_pos_target = None

            self.tracer_spec = None
            self.tracer_loop = None
            self.tracer_adapter = None

            super().__init__(cfg, render_mode=render_mode, **kwargs)

        def _setup_scene(self):
            self.robot = Articulation(self.cfg.robot)
            self.scene.articulations["robot"] = self.robot

            ground_cfg = sim_utils.GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                )
            )
            ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions(global_prim_paths=["/World/defaultGroundPlane"])

            light_cfg = sim_utils.DomeLightCfg(
                intensity=2000.0,
                color=(0.75, 0.75, 0.75),
            )
            light_cfg.func("/World/Light", light_cfg)

        def _init_tracer_if_needed(self):
            if self.tracer_adapter is not None:
                return

            self.tracer_spec = TracerTaskSpec.from_json(
                tracer_config_path("configs/isaac_lab/tracer_task_v0.json")
            )

            self.tracer_loop = TracerHighLevelLoop(
                spec=self.tracer_spec,
                num_envs=self.num_envs,
                device=str(self.device),
            )

            goal_xy = torch.zeros(self.num_envs, 2, device=self.device)

            # Use env origins if available, so each cloned env gets a local relative goal.
            try:
                origins = self.scene.env_origins.to(self.device)
                goal_xy[:, 0] = origins[:, 0] + self.cfg.tracer_goal_x
                goal_xy[:, 1] = origins[:, 1] + self.cfg.tracer_goal_y
            except Exception:
                goal_xy[:, 0] = self.cfg.tracer_goal_x
                goal_xy[:, 1] = self.cfg.tracer_goal_y

            self.tracer_loop.set_goal_xy(goal_xy)

            default_joint_pos = self.robot.data.default_joint_pos.detach().clone()

            highlevel_loop = self.tracer_loop

            # Avoid accessing config/spec dynamic attributes inside IsaacLab reset path.
            # For this smoke test, use explicit constants.
            success_tolerance = 0.30
            residual_scale = 0.10


            adapter = TracerStepAdapter(
                highlevel_loop=highlevel_loop,
                default_joint_pos=default_joint_pos,
                success_tolerance=success_tolerance,
                residual_scale=residual_scale,
            )
            self.tracer_adapter = adapter

            self._previous_action = torch.zeros(
                self.num_envs,
                self.cfg.action_space,
                device=self.device,
            )

        def _make_tracer_obs_dict(self):
            root_pos_w = self.robot.data.root_pos_w
            root_quat_w = self.robot.data.root_quat_w

            roll, pitch, yaw = quat_wxyz_to_rpy(root_quat_w)

            root_lin_vel_b = self.robot.data.root_lin_vel_b
            root_ang_vel_b = self.robot.data.root_ang_vel_b

            try:
                projected_gravity = self.robot.data.projected_gravity_b
            except Exception:
                projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
                projected_gravity[:, 2] = -1.0

            height_scan = torch.zeros(self.num_envs, 10, device=self.device)

            if self._previous_action is None:
                self._previous_action = torch.zeros(
                    self.num_envs,
                    self.cfg.action_space,
                    device=self.device,
                )

            return {
                "base_xy": root_pos_w[:, 0:2],
                "base_yaw": yaw,
                "base_lin_vel_body": root_lin_vel_b,
                "base_ang_vel_body": root_ang_vel_b,
                "base_yaw_rate": root_ang_vel_b[:, 2:3],
                "projected_gravity": projected_gravity,
                "height_scan": height_scan,
                "base_height": root_pos_w[:, 2:3],
                "roll": roll,
                "pitch": pitch,
                "joint_pos": self.robot.data.joint_pos,
                "joint_vel": self.robot.data.joint_vel,
                "previous_action": self._previous_action,
            }

        def _pre_physics_step(self, actions):
            self._init_tracer_if_needed()

            self._actions = torch.clamp(actions, -1.0, 1.0)

            obs_dict = self._make_tracer_obs_dict()
            adapter_out = self.tracer_adapter.step(obs_dict, self._actions)

            self._policy_obs = adapter_out.policy_obs
            self._joint_pos_target = adapter_out.joint_pos_target
            self._reward = adapter_out.reward
            self._terminated = adapter_out.done
            self._previous_action = self._actions.detach()

            self.extras["tracer/goal_distance_mean"] = adapter_out.tracer.goal_features[:, 0].mean()
            self.extras["tracer/vx_cmd_mean"] = adapter_out.tracer.tracer_cmd[:, 0].mean()
            self.extras["tracer/yaw_cmd_mean"] = adapter_out.tracer.tracer_cmd[:, 1].mean()
            self.extras["tracer/rho_mean"] = adapter_out.tracer.rho.mean()
            self.extras["tracer/sigma_mean"] = adapter_out.tracer.sigma.mean()
            self.extras["tracer/beta_v_mean"] = adapter_out.tracer.beta[:, 0].mean()
            self.extras["tracer/beta_s_mean"] = adapter_out.tracer.beta[:, 1].mean()
            self.extras["tracer/beta_e_mean"] = adapter_out.tracer.beta[:, 2].mean()

        def _apply_action(self):
            if self._joint_pos_target is None:
                target = self.robot.data.default_joint_pos
            else:
                target = self._joint_pos_target

            self.robot.set_joint_position_target(target)

        def _get_observations(self):
            if self._policy_obs is None:
                self._policy_obs = torch.zeros(
                    self.num_envs,
                    self.cfg.observation_space,
                    device=self.device,
                )

            return {"policy": self._policy_obs}

        def _get_rewards(self):
            if self._reward is None:
                return torch.zeros(self.num_envs, device=self.device)

            return self._reward

        def _get_dones(self):
            if self._terminated is None:
                terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            else:
                terminated = self._terminated

            root_height = self.robot.data.root_pos_w[:, 2]
            height_fail = root_height < 0.18
            terminated = torch.logical_or(terminated, height_fail)

            truncated = self.episode_length_buf >= self.max_episode_length - 1
            return terminated, truncated

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)

            self._init_tracer_if_needed()

            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

            self.tracer_adapter.reset(env_ids)
            self._previous_action[env_ids] = 0.0

            if self._policy_obs is not None:
                self._policy_obs[env_ids] = 0.0

            self._reward = None
            self._terminated = None
            self._joint_pos_target = None

    return TracerA1AdapterEnv, TracerA1AdapterEnvCfg
