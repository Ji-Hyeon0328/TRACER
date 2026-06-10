import torch


def make_tracer_a1_spawn_env_class():
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.assets import Articulation
    from isaaclab.utils import configclass
    import isaaclab.sim as sim_utils
    from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

    @configclass
    class TracerA1SpawnEnvCfg(DirectRLEnvCfg):
        episode_length_s = 2.0
        decimation = 1

        action_space = 12
        observation_space = 58
        state_space = 0

        sim: SimulationCfg = SimulationCfg(
            dt=0.02,
            render_interval=decimation,
        )

        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=4,
            env_spacing=2.5,
        )

        robot = UNITREE_A1_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
        )

    class TracerA1SpawnEnv(DirectRLEnv):
        cfg: TracerA1SpawnEnvCfg

        def __init__(self, cfg: TracerA1SpawnEnvCfg, render_mode=None, **kwargs):
            self.robot = None
            self._actions = None
            self._policy_obs = None
            super().__init__(cfg, render_mode=render_mode, **kwargs)

        def _setup_scene(self):
            self.robot = Articulation(self.cfg.robot)
            self.scene.articulations["robot"] = self.robot

            ground_cfg = sim_utils.GroundPlaneCfg()
            ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions(global_prim_paths=["/World/defaultGroundPlane"])

            light_cfg = sim_utils.DomeLightCfg(
                intensity=2000.0,
                color=(0.75, 0.75, 0.75),
            )
            light_cfg.func("/World/Light", light_cfg)

        def _pre_physics_step(self, actions):
            self._actions = torch.clamp(actions, -1.0, 1.0)

        def _apply_action(self):
            # For M32, hold default joint positions with a tiny residual action.
            default_pos = self.robot.data.default_joint_pos
            target = default_pos + 0.05 * self._actions
            self.robot.set_joint_position_target(target)

        def _get_observations(self):
            if self._policy_obs is None:
                self._policy_obs = torch.zeros(
                    self.num_envs,
                    self.cfg.observation_space,
                    device=self.device,
                )

            root_pos_w = self.robot.data.root_pos_w
            root_quat_w = self.robot.data.root_quat_w
            root_lin_vel_b = self.robot.data.root_lin_vel_b
            root_ang_vel_b = self.robot.data.root_ang_vel_b
            joint_pos = self.robot.data.joint_pos
            joint_vel = self.robot.data.joint_vel

            self._policy_obs.zero_()
            self._policy_obs[:, 0:3] = root_lin_vel_b
            self._policy_obs[:, 3:6] = root_ang_vel_b
            self._policy_obs[:, 6:18] = joint_pos
            self._policy_obs[:, 18:30] = joint_vel
            if self._actions is not None:
                self._policy_obs[:, 30:42] = self._actions

            # Store a few values for debug.
            self.extras["debug/root_height_mean"] = root_pos_w[:, 2].mean()
            self.extras["debug/root_quat_w_mean"] = root_quat_w.mean()
            self.extras["debug/joint_pos_mean"] = joint_pos.mean()

            return {"policy": self._policy_obs}

        def _get_rewards(self):
            root_height = self.robot.data.root_pos_w[:, 2]
            alive = (root_height > 0.20).float()
            action_penalty = torch.zeros_like(root_height)
            if self._actions is not None:
                action_penalty = torch.sum(self._actions * self._actions, dim=-1)
            return alive - 0.01 * action_penalty

        def _get_dones(self):
            root_height = self.robot.data.root_pos_w[:, 2]
            terminated = root_height < 0.18
            time_out = self.episode_length_buf >= self.max_episode_length - 1
            return terminated, time_out

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)

            if self._policy_obs is None:
                self._policy_obs = torch.zeros(
                    self.num_envs,
                    self.cfg.observation_space,
                    device=self.device,
                )

            if env_ids is None:
                env_ids = self.robot._ALL_INDICES

            self._policy_obs[env_ids] = 0.0

    return TracerA1SpawnEnv, TracerA1SpawnEnvCfg
