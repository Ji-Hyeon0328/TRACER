import torch


def make_tracer_a1_stand_env_class():
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.assets import Articulation
    from isaaclab.utils import configclass
    import isaaclab.sim as sim_utils
    from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO2_CFG

    @configclass
    class TracerA1StandEnvCfg(DirectRLEnvCfg):
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

        robot_name = "a1"

        robot = UNITREE_A1_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
        )

        # M34 stand-hold tuning.
        stand_stiffness = 60.0
        stand_damping = 2.0
        residual_scale = 0.0
        disable_termination = False
        use_explicit_stand_pose = False
        stand_base_height = 0.36
        stand_hip_abs = 0.10
        stand_front_thigh = 0.80
        stand_rear_thigh = 1.00
        stand_calf = -1.50
        freeze_when_fallen = False
        freeze_height = 0.22

    class TracerA1StandEnv(DirectRLEnv):
        cfg: TracerA1StandEnvCfg

        def __init__(self, cfg: TracerA1StandEnvCfg, render_mode=None, **kwargs):
            self.robot = None
            self._actions = None
            self._policy_obs = None
            self._stand_target = None
            super().__init__(cfg, render_mode=render_mode, **kwargs)

        def _setup_scene(self):
            # Select robot asset.
            if self.cfg.robot_name.lower() == "go2":
                robot_cfg = UNITREE_GO2_CFG.replace(
                    prim_path="/World/envs/env_.*/Robot",
                ).copy()
            elif self.cfg.robot_name.lower() == "a1":
                robot_cfg = UNITREE_A1_CFG.replace(
                    prim_path="/World/envs/env_.*/Robot",
                ).copy()
            else:
                raise ValueError(f"Unsupported robot_name: {self.cfg.robot_name}")

            print("[M37] robot_name:", self.cfg.robot_name, flush=True)

            try:
                actuator = robot_cfg.actuators["base_legs"]
                actuator.stiffness = self.cfg.stand_stiffness
                actuator.damping = self.cfg.stand_damping
                print(
                    "[M34] override actuator gains:",
                    "stiffness=", actuator.stiffness,
                    "damping=", actuator.damping,
                    flush=True,
                )
            except Exception as e:
                print("[M34][WARN] could not override actuator gains:", e, flush=True)

            self.robot = Articulation(robot_cfg)
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

        def _build_stand_target(self, env_ids=None):
            if not self.cfg.use_explicit_stand_pose:
                if env_ids is None:
                    return self.robot.data.default_joint_pos.detach().clone()
                return self.robot.data.default_joint_pos[env_ids].detach().clone()

            if env_ids is None:
                target = self.robot.data.default_joint_pos.detach().clone()
            else:
                target = self.robot.data.default_joint_pos[env_ids].detach().clone()

            # Joint order:
            # FL_hip, FR_hip, RL_hip, RR_hip,
            # FL_thigh, FR_thigh, RL_thigh, RR_thigh,
            # FL_calf, FR_calf, RL_calf, RR_calf
            target[:, 0] = self.cfg.stand_hip_abs
            target[:, 1] = -self.cfg.stand_hip_abs
            target[:, 2] = self.cfg.stand_hip_abs
            target[:, 3] = -self.cfg.stand_hip_abs

            target[:, 4] = self.cfg.stand_front_thigh
            target[:, 5] = self.cfg.stand_front_thigh
            target[:, 6] = self.cfg.stand_rear_thigh
            target[:, 7] = self.cfg.stand_rear_thigh

            target[:, 8:12] = self.cfg.stand_calf
            return target

        def _pre_physics_step(self, actions):
            self._actions = torch.clamp(actions, -1.0, 1.0)

            if self._stand_target is None:
                self._stand_target = self._build_stand_target()

        def _apply_action(self):
            if self._stand_target is None:
                target = self.robot.data.default_joint_pos
            else:
                target = self._stand_target.clone()

            # Debug safety: once the robot has fallen, stop fighting contact constraints.
            # This prevents high-frequency leg oscillation while lying on the ground.
            if self.cfg.freeze_when_fallen:
                root_height = self.robot.data.root_pos_w[:, 2]
                fallen = root_height < self.cfg.freeze_height
                if torch.any(fallen):
                    target[fallen] = self.robot.data.joint_pos[fallen].detach()

            self.robot.set_joint_position_target(target)

        def _get_observations(self):
            if self._policy_obs is None:
                self._policy_obs = torch.zeros(
                    self.num_envs,
                    self.cfg.observation_space,
                    device=self.device,
                )

            self._policy_obs.zero_()
            self._policy_obs[:, 0:3] = self.robot.data.root_lin_vel_b
            self._policy_obs[:, 3:6] = self.robot.data.root_ang_vel_b
            self._policy_obs[:, 6:18] = self.robot.data.joint_pos
            self._policy_obs[:, 18:30] = self.robot.data.joint_vel
            if self._actions is not None:
                self._policy_obs[:, 30:42] = self._actions

            self.extras["stand/root_height_mean"] = self.robot.data.root_pos_w[:, 2].mean()
            self.extras["stand/joint_pos_mean"] = self.robot.data.joint_pos.mean()
            self.extras["stand/joint_vel_abs_mean"] = self.robot.data.joint_vel.abs().mean()

            return {"policy": self._policy_obs}

        def _get_rewards(self):
            root_height = self.robot.data.root_pos_w[:, 2]
            height_reward = torch.exp(-20.0 * torch.square(root_height - 0.40))
            vel_penalty = 0.01 * self.robot.data.joint_vel.abs().mean(dim=-1)
            return height_reward - vel_penalty

        def _get_dones(self):
            if self.cfg.disable_termination:
                terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                return terminated, truncated

            root_height = self.robot.data.root_pos_w[:, 2]
            terminated = root_height < 0.18
            truncated = self.episode_length_buf >= self.max_episode_length - 1
            return terminated, truncated

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)

            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

            # Explicitly reset the robot state to remove initial joint-target mismatch.
            root_state = self.robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += self.scene.env_origins[env_ids]
            root_state[:, 2] = self.scene.env_origins[env_ids, 2] + self.cfg.stand_base_height

            joint_pos = self._build_stand_target(env_ids)
            joint_vel = torch.zeros_like(joint_pos)

            self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
            self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

            if self._policy_obs is not None:
                self._policy_obs[env_ids] = 0.0

            # Hold exactly the same joint state we just wrote.
            self._stand_target = self._build_stand_target()

    return TracerA1StandEnv, TracerA1StandEnvCfg
