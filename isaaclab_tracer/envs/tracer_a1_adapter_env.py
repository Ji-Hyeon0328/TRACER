import math

import torch

from isaaclab_tracer.controllers.a1_gait_core import A1GaitCore, A1GaitCoreCfg
from isaaclab_tracer.controllers.a1_kinematics_torch import A1KinematicsTorch
from isaaclab_tracer.envs.tracer_task_spec import TracerTaskSpec
from isaaclab_tracer.envs.tracer_highlevel_loop import TracerHighLevelLoop
from isaaclab_tracer.envs.tracer_step_adapter import TracerStepAdapter
from isaaclab_tracer.utils.paths import tracer_config_path
from isaaclab_tracer.utils.lowlevel_udp_client import LowLevelUdpClient


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

        # TRACER action semantics.
        #
        # joint_position_residual:
        #   existing behavior. PPO action is a 12D joint residual.
        #
        # meta_gait_theta:
        #   PPO action is a 6D structured meta-gait plan theta.
        #   The env maps theta -> gait command / gait proxy -> IK joint targets.
        action_type = "joint_position_residual"
        meta_theta_dim = 6
        meta_base_vx = 0.10
        meta_vx_scale = 0.07
        meta_min_vx = 0.00
        meta_max_vx = 0.18
        meta_yaw_scale = 0.15
        meta_body_height_delta = 0.04
        meta_clearance_delta = 0.08
        meta_debug = False

        # Do not start the gait oscillator when commanded vx is near zero.
        # This prevents zero-command stepping drift during diagnostics/training.
        meta_movement_vx_threshold = 0.02

        # Isaac/A1 gait-core body-frame x convention is opposite to TRACER's
        # external forward-vx convention. Keep TRACER vx positive-forward,
        # but flip it before sending into A1GaitCore.
        meta_gait_x_sign = -1.0

        # Small stance-foot backward drift in body frame.
        # This creates forward propulsion while swing legs reposition.
        meta_stance_push_gain = 1.0

        # Blend stance-leg IK target with default joint pose.
        # 0.0 = old behavior, 1.0 = full stance IK target.
        meta_stance_ik_blend = 0.6
        # PPO safety defaults for early meta-gait training.
        # V0 trains only the forward velocity residual. Other theta dimensions
        # are kept frozen until the gait basin is robust enough.
        meta_train_vx_only = True
        meta_theta_smoothing_alpha = 0.90
        meta_done_min_height = 0.22
        meta_done_lateral_dist = 0.35
        meta_done_backward_dist = 0.25
        # PPO reward defaults for meta_gait_theta.
        # These are intentionally conservative: first learn to improve
        # forward progress without destroying the stable weak-forward prior.
        meta_reward_forward_vel = 4.0
        meta_reward_forward_progress = 2.0
        meta_penalty_lateral_vel = 0.5
        meta_penalty_yaw_rate = 0.15
        meta_penalty_ang_vel = 0.05
        meta_penalty_height = 4.0
        meta_target_height = 0.285
        meta_penalty_low_height = 8.0
        meta_safe_min_height = 0.255
        meta_penalty_joint_vel = 0.002
        meta_penalty_torque = 0.0005
        meta_penalty_action = 0.02
        meta_penalty_action_rate = 0.02
        meta_fall_penalty = 10.0

        # TRACER beta-weighted reward scaffold.
        #
        # beta_velocity:
        #   how strongly the policy should care about forward progress.
        # beta_stability:
        #   how strongly the policy should care about posture, lateral drift,
        #   yaw/roll/pitch motion, and low-height safety.
        # beta_energy:
        #   how strongly the policy should care about joint velocity / torque cost.
        # beta_clearance:
        #   optional terrain/style reward gate. Default is zero so flat V0 is unchanged.
        #
        # Later, Objective Selector will predict these beta values from terrain context.
        meta_beta_velocity = 1.0
        meta_beta_stability = 1.0
        meta_beta_energy = 1.0
        meta_beta_clearance = 0.0
        meta_reward_clearance = 0.0
        meta_use_beta_reward = False

        # TRACER terrain/material scaffold.
        #
        # This is the first physical hook for terrain-aware gait learning.
        # For now it controls the ground-plane material/friction. Later this can
        # be extended to heightfield/mesh rough terrain.
        terrain_preset = "flat"
        terrain_static_friction = 1.0
        terrain_dynamic_friction = 1.0
        terrain_restitution = 0.0

        # Debug/diagnostic option:
        # Ignore TracerStepAdapter done signals and only use env-level safety
        # termination. Useful for standing/controller diagnostics.
        ignore_adapter_done = False

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

        # Reset height should be consistent with default_joint_pos FK.
        # For current A1 default pose, mean foot z is about -0.299 m.
        # Spawning at 0.42 m leaves the feet floating and causes drop/termination.
        reset_root_height = 0.385

        residual_scale = 0.0
        hold_default_pose = False
        use_nominal_gait = False
        gait_cmd_x = 0.0
        gait_cmd_y = 0.0
        gait_pattern = "trot"
        gait_warmup_steps = 80
        gait_counter_speed = 2.0
        gait_clearance2 = 0.03
        gait_foot_delta_x_limit = 0.05
        gait_foot_delta_y_limit = 0.03

        # External low-level controller bridge.
        # This keeps Isaac Lab in conda Python and talks to ROS2 through UDP.
        use_external_lowlevel = False
        external_lowlevel_host = "127.0.0.1"
        external_lowlevel_state_port = 50100
        external_lowlevel_cmd_port = 50101
        external_lowlevel_timeout_s = 0.20
        external_lowlevel_env_index = 0

        # Actuator PD gains for external low-level / torque-FF diagnostics.
        # Lowering these reduces default-position PD dominance in mode=3.
        adapter_actuator_stiffness = 60.0
        adapter_actuator_damping = 2.0
        adapter_actuator_effort_limit = 120.0

    class TracerA1AdapterEnv(DirectRLEnv):
        cfg: TracerA1AdapterEnvCfg

        def __init__(self, cfg: TracerA1AdapterEnvCfg, render_mode=None, **kwargs):
            self.robot = None
            self._residual_scale = float(cfg.residual_scale)
            self._hold_default_pose = bool(cfg.hold_default_pose)

            self._actions = None
            self._previous_action = None
            self._previous_adapter_action = None
            self._policy_obs = None
            self._reward = None
            self._terminated = None
            self._truncated = None
            self._joint_pos_target = None
            self._joint_effort_target = None

            self.tracer_spec = None
            self.tracer_loop = None
            self.tracer_adapter = None
            self.gait_core = None
            self.a1_kin = None
            self.lowlevel_udp = None

            super().__init__(cfg, render_mode=render_mode, **kwargs)

        def _setup_scene(self):
            try:
                actuator = self.cfg.robot.actuators["base_legs"]
                actuator.stiffness = float(self.cfg.adapter_actuator_stiffness)
                actuator.damping = float(self.cfg.adapter_actuator_damping)
                print(
                    "[A1Adapter] override actuator gains:",
                    "stiffness=", actuator.stiffness,
                    "damping=", actuator.damping,
                    flush=True,
                )
            except Exception as e:
                print("[A1Adapter][WARN] could not override actuator gains:", e, flush=True)

            self.robot = Articulation(self.cfg.robot)
            self.scene.articulations["robot"] = self.robot

            terrain_preset = str(getattr(self.cfg, "terrain_preset", "flat")).lower()
            terrain_static_friction = float(getattr(self.cfg, "terrain_static_friction", 1.0))
            terrain_dynamic_friction = float(getattr(self.cfg, "terrain_dynamic_friction", 1.0))
            terrain_restitution = float(getattr(self.cfg, "terrain_restitution", 0.0))

            print(
                "[A1Adapter] terrain material:",
                "preset=", terrain_preset,
                "static_friction=", terrain_static_friction,
                "dynamic_friction=", terrain_dynamic_friction,
                "restitution=", terrain_restitution,
                flush=True,
            )

            ground_cfg = sim_utils.GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=terrain_static_friction,
                    dynamic_friction=terrain_dynamic_friction,
                    restitution=terrain_restitution,
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

            # Avoid accessing spec dynamic attributes inside IsaacLab reset path.
            # Cache cfg values in __init__, then use the cached value here.
            success_tolerance = 0.30
            residual_scale = self._residual_scale

            adapter = TracerStepAdapter(
                highlevel_loop=highlevel_loop,
                default_joint_pos=default_joint_pos,
                success_tolerance=success_tolerance,
                residual_scale=residual_scale,
            )
            self.tracer_adapter = adapter

            gait_cfg = A1GaitCoreCfg()
            gait_cfg.foot_swing_clearance2 = float(self.cfg.gait_clearance2)
            gait_cfg.gait_counter_speed = (float(self.cfg.gait_counter_speed),) * 4
            gait_cfg.foot_delta_x_limit = float(getattr(self.cfg, "gait_foot_delta_x_limit", gait_cfg.foot_delta_x_limit))
            gait_cfg.foot_delta_y_limit = float(getattr(self.cfg, "gait_foot_delta_y_limit", gait_cfg.foot_delta_y_limit))
            if str(self.cfg.gait_pattern).lower() == "crawl":
                # Crawl-like schedule:
                # - 75% stance, 25% swing
                # - one leg swings at a time
                # - order with this initialization is roughly RR -> RL -> FR -> FL.
                # This avoids immediately lifting FR/RL together as in trot.
                gait_cfg.counter_per_gait = 240.0
                gait_cfg.counter_per_swing = 180.0
                gait_cfg.init_gait_counter = (0.0, 60.0, 120.0, 180.0)
            elif str(self.cfg.gait_pattern).lower() == "trot":
                # Baseline diagonal trot.
                gait_cfg.counter_per_gait = 240.0
                gait_cfg.counter_per_swing = 120.0
                gait_cfg.init_gait_counter = (0.0, 120.0, 120.0, 0.0)
            else:
                raise ValueError(f"Unknown gait_pattern: {self.cfg.gait_pattern}")
            gait_pattern = "trot"
            self.gait_core = A1GaitCore(
                num_envs=self.num_envs,
                device=self.device,
                cfg=gait_cfg,
            )
            self.a1_kin = A1KinematicsTorch(device=self.device)

            default_foot = self.a1_kin.fk_isaac(self.robot.data.default_joint_pos.detach())
            self.gait_core.set_default_foot_pos(default_foot)

            self._previous_action = torch.zeros(
                self.num_envs,
                self.cfg.action_space,
                device=self.device,
            )
            self._previous_adapter_action = torch.zeros(
                self.num_envs,
                12,
                device=self.device,
            )

        def _make_tracer_obs_dict(self):
            root_pos_w = self.robot.data.root_pos_w
            root_quat_w = self.robot.data.root_quat_w

            roll, pitch, yaw = quat_wxyz_to_rpy(root_quat_w)

            try:
                projected_gravity = self.robot.data.projected_gravity_b
            except Exception:
                projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
                projected_gravity[:, 2] = -1.0

            height_scan = torch.zeros(self.num_envs, 1, device=self.device)

            obs = {
                "base_xy": root_pos_w[:, 0:2],
                "base_pos_xy": root_pos_w[:, 0:2],
                "base_height": root_pos_w[:, 2:3],
                "height_scan": height_scan,
                "base_quat_w": root_quat_w,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "base_yaw": yaw,
                "base_lin_vel_body": self.robot.data.root_lin_vel_b,
                "base_ang_vel_body": self.robot.data.root_ang_vel_b,
                "base_yaw_rate": self.robot.data.root_ang_vel_b[:, 2:3],
                "base_lin_vel": self.robot.data.root_lin_vel_b,
                "base_ang_vel": self.robot.data.root_ang_vel_b,
                "projected_gravity": projected_gravity,
                "joint_pos": self.robot.data.joint_pos,
                "joint_vel": self.robot.data.joint_vel,
            }

            return obs


        def _init_lowlevel_udp_if_needed(self):
            if self.lowlevel_udp is not None:
                return
            self.lowlevel_udp = LowLevelUdpClient(
                host=str(self.cfg.external_lowlevel_host),
                state_send_port=int(self.cfg.external_lowlevel_state_port),
                cmd_recv_port=int(self.cfg.external_lowlevel_cmd_port),
            )

        def _build_external_robot_state(self):
            """Build /tracer/robot_state layout for one selected env.

            Layout:
              [0]      stamp_sec
              [1:4]    base_pos xyz
              [4:8]    base_quat xyzw
              [8:11]   base_lin_vel xyz
              [11:14]  base_ang_vel xyz
              [14:26]  joint_pos[12]
              [26:38]  joint_vel[12]
              [38:42]  foot_contact[4]
            """
            env_id = int(self.cfg.external_lowlevel_env_index)
            env_id = max(0, min(env_id, self.num_envs - 1))

            root_pos = self.robot.data.root_pos_w[env_id].detach().cpu().tolist()

            # IsaacLab root_quat_w is usually [w, x, y, z].
            # TRACER external interface uses xyzw.
            q_wxyz = self.robot.data.root_quat_w[env_id].detach().cpu().tolist()
            q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]

            # Use body-frame velocity for now, matching the controller command convention.
            base_lin_vel = self.robot.data.root_lin_vel_b[env_id].detach().cpu().tolist()
            base_ang_vel = self.robot.data.root_ang_vel_b[env_id].detach().cpu().tolist()

            joint_pos = self.robot.data.joint_pos[env_id].detach().cpu().tolist()
            joint_vel = self.robot.data.joint_vel[env_id].detach().cpu().tolist()

            # v0: no real contact sensor in this bridge yet. Treat as all contact for safe standing.
            foot_contact = [1.0, 1.0, 1.0, 1.0]

            stamp_sec = float(self.common_step_counter) * float(self.cfg.sim.dt) * float(self.cfg.decimation)

            return (
                [stamp_sec]
                + root_pos
                + q_xyzw
                + base_lin_vel
                + base_ang_vel
                + joint_pos
                + joint_vel
                + foot_contact
            )

        def _apply_external_lowlevel_if_available(self):
            """Send Isaac state to UDP bridge and apply fresh low-level command.

            Returns:
                True if external command branch handled this step.
                False if caller should use the normal internal controller branch.
            """
            if not bool(getattr(self.cfg, "use_external_lowlevel", False)):
                return False

            self._init_lowlevel_udp_if_needed()

            # Publish selected env state to ROS2 sidecar.
            try:
                self.lowlevel_udp.send_robot_state(self._build_external_robot_state())
            except Exception:
                # Do not crash Isaac because of bridge packet failure.
                pass

            cmd = self.lowlevel_udp.get_latest_cmd(
                timeout_s=float(self.cfg.external_lowlevel_timeout_s)
            )

            # Watchdog fallback: if command is missing/stale, hold current pose with zero torque.
            if cmd is None:
                self._joint_pos_target = self.robot.data.joint_pos.detach().clone()
                self._joint_effort_target = torch.zeros_like(self.robot.data.default_joint_pos)
                return True

            mode = int(round(float(cmd[0])))

            q_target_one = torch.tensor(
                cmd[1:13],
                dtype=self.robot.data.joint_pos.dtype,
                device=self.device,
            ).view(1, 12)
            dq_target_one = torch.tensor(
                cmd[13:25],
                dtype=self.robot.data.joint_pos.dtype,
                device=self.device,
            ).view(1, 12)
            tau_ff_one = torch.tensor(
                cmd[49:61],
                dtype=self.robot.data.joint_pos.dtype,
                device=self.device,
            ).view(1, 12)

            q_target = q_target_one.repeat(self.num_envs, 1)
            tau_ff = tau_ff_one.repeat(self.num_envs, 1)

            if mode == 1:
                # Joint position target mode.
                self._joint_pos_target = q_target
                self._joint_effort_target = torch.zeros_like(self.robot.data.default_joint_pos)
            elif mode == 2:
                # Torque mode. Hold current position so position drive does not introduce new motion.
                self._joint_pos_target = self.robot.data.joint_pos.detach().clone()
                self._joint_effort_target = tau_ff
            elif mode == 3:
                # Hybrid mode: position target + feedforward torque.
                self._joint_pos_target = q_target
                self._joint_effort_target = tau_ff
            else:
                # Unknown mode: safe hold.
                self._joint_pos_target = self.robot.data.joint_pos.detach().clone()
                self._joint_effort_target = torch.zeros_like(self.robot.data.default_joint_pos)

            return True


        def _pre_physics_step(self, actions):
            self._actions = actions.detach().clone()

            action_type = str(getattr(self.cfg, "action_type", "joint_position_residual")).lower()
            is_meta_theta = action_type == "meta_gait_theta"

            if self._previous_action is None:
                self._previous_action = torch.zeros_like(self._actions)

            if self._previous_adapter_action is None:
                self._previous_adapter_action = torch.zeros(self.num_envs, 12, device=self.device)

            obs_dict = self._make_tracer_obs_dict()

            # TracerStepAdapter and policy_obs_v0 still expect a 12D low-level previous action.
            # In meta-gait mode, PPO action is theta, so feed a safe 12D zero adapter action
            # into the legacy adapter/reward path.
            if is_meta_theta:
                # TRACER meta-gait curriculum support:
                # Map low-dimensional RL actions into selected full theta indices.
                # Example: active=[0] means action[0] -> theta[0].
                #          active=[0, 3] with scales=[1.0, 0.25] means
                #          action[0] -> theta[0] * 1.0, action[1] -> theta[3] * 0.25.
                theta_dim = int(getattr(self.cfg, "meta_theta_dim", 6))
                active_indices = getattr(self.cfg, "meta_active_theta_indices", None)
                if active_indices is not None:
                    active_indices = [int(idx) for idx in active_indices]
                
                if active_indices:
                    if actions.shape[-1] != len(active_indices):
                        raise ValueError(
                            f"meta_active_theta_indices expects action dim {len(active_indices)}, got {actions.shape[-1]}"
                        )
                
                    active_scales = getattr(self.cfg, "meta_active_theta_scales", None)
                    if active_scales is None:
                        active_scales = [1.0 for _ in active_indices]
                    active_scales = [float(s) for s in active_scales]
                
                    if len(active_scales) != len(active_indices):
                        raise ValueError(
                            f"meta_active_theta_scales length {len(active_scales)} must match "
                            f"meta_active_theta_indices length {len(active_indices)}"
                        )
                
                    full_actions = torch.zeros(
                        (actions.shape[0], theta_dim),
                        device=actions.device,
                        dtype=actions.dtype,
                    )
                    for src_i, theta_i in enumerate(active_indices):
                        if theta_i < 0 or theta_i >= theta_dim:
                            raise ValueError(f"Invalid active theta index {theta_i} for theta_dim={theta_dim}")
                        full_actions[:, theta_i] = actions[:, src_i] * active_scales[src_i]
                    actions = full_actions
                else:
                    if actions.shape[-1] < theta_dim:
                        padded_actions = torch.zeros(
                            (actions.shape[0], theta_dim),
                            device=actions.device,
                            dtype=actions.dtype,
                        )
                        padded_actions[:, : actions.shape[-1]] = actions
                        actions = padded_actions
                    elif actions.shape[-1] > theta_dim:
                        raise ValueError(
                            f"meta_gait_theta expects action dim <= {theta_dim}, got {actions.shape[-1]}"
                        )
                
                # Keep internal cached action consistent with the full meta-gait theta.
                self._actions = actions
                
                if self._actions.shape[-1] != int(getattr(self.cfg, "meta_theta_dim", 6)):
                    raise ValueError(
                        f"meta_gait_theta expects action dim {int(getattr(self.cfg, 'meta_theta_dim', 6))}, "
                        f"got {self._actions.shape[-1]}"
                    )
                adapter_action = torch.zeros(self.num_envs, 12, device=self.device)
                obs_dict["previous_action"] = self._previous_adapter_action
            else:
                adapter_action = self._actions
                obs_dict["previous_action"] = self._previous_action

            out = self.tracer_adapter.step(
                obs_dict,
                adapter_action,
            )

            self._policy_obs = out.policy_obs
            self._reward = out.reward
            if bool(getattr(self.cfg, "ignore_adapter_done", False)):
                self._terminated = torch.zeros_like(out.done, dtype=torch.bool)
            else:
                self._terminated = out.done
            self._truncated = torch.zeros_like(out.done, dtype=torch.bool)

            if self._apply_external_lowlevel_if_available():
                self._previous_action = self._actions.detach().clone()
                self._previous_adapter_action = adapter_action.detach().clone()
                return

            if is_meta_theta:
                # Structured TRACER meta-gait action.
                #
                # theta[0] -> forward velocity command around a positive walking prior
                # theta[1] -> lateral/yaw proxy, reserved for later
                # theta[2] -> body-height proxy through foot z offset
                # theta[3] -> extra swing clearance
                # theta[4] -> gait-speed/period proxy, reserved for later
                # theta[5] -> residual/gain proxy, reserved for later
                theta_raw = torch.clamp(self._actions, -1.0, 1.0)

                # Early PPO safety:
                # Train only theta[0] = forward velocity residual.
                # Keep body height / clearance / other meta dimensions frozen
                # until the stable gait basin is wider.
                if bool(getattr(self.cfg, "meta_train_vx_only", True)):
                    theta_raw = theta_raw.clone()
                    if theta_raw.shape[1] > 1:
                        theta_raw[:, 1:] = 0.0

                # Low-pass filter meta actions. Direct per-step random theta
                # is too aggressive for the current IK gait target.
                alpha = float(getattr(self.cfg, "meta_theta_smoothing_alpha", 0.90))
                alpha = max(0.0, min(0.99, alpha))
                if not hasattr(self, "_meta_theta_smooth") or self._meta_theta_smooth is None:
                    self._meta_theta_smooth = torch.zeros_like(theta_raw)
                if self._meta_theta_smooth.shape != theta_raw.shape:
                    self._meta_theta_smooth = torch.zeros_like(theta_raw)
                self._meta_theta_smooth = alpha * self._meta_theta_smooth + (1.0 - alpha) * theta_raw
                theta = self._meta_theta_smooth

                # Store the effective smoothed theta for reward terms / diagnostics.
                # This is the full 6D TRACER theta after active-index mapping,
                # optional per-index scaling, vx-only masking, and smoothing.
                self._last_meta_theta = theta.detach().clone()

                self._joint_effort_target = torch.zeros_like(self.robot.data.default_joint_pos)

                foot_cur = self.a1_kin.fk_isaac(self.robot.data.joint_pos.detach())

                cmd_body = torch.zeros(self.num_envs, 3, device=self.device)
                vx_cmd = (
                    float(getattr(self.cfg, "meta_base_vx", 0.18))
                    + float(getattr(self.cfg, "meta_vx_scale", 0.12)) * theta[:, 0]
                )
                vx_cmd = torch.clamp(
                    vx_cmd,
                    float(getattr(self.cfg, "meta_min_vx", 0.03)),
                    float(getattr(self.cfg, "meta_max_vx", 0.30)),
                )
                cmd_body[:, 0] = float(getattr(self.cfg, "meta_gait_x_sign", -1.0)) * vx_cmd

                # V0 stability: no lateral command proxy yet.
                # Yaw/lateral gait modulation will be added after forward gait is stable.
                cmd_body[:, 1] = 0.0

                warmup_done = bool(
                    torch.min(self.episode_length_buf).item() >= int(self.cfg.gait_warmup_steps)
                )
                vx_active = bool(
                    torch.max(torch.abs(vx_cmd)).item()
                    > float(getattr(self.cfg, "meta_movement_vx_threshold", 0.02))
                )
                movement_mode = warmup_done and vx_active

                gait_out = self.gait_core.step(
                    foot_pos_cur_rel=foot_cur,
                    root_lin_vel_body=self.robot.data.root_lin_vel_b.detach(),
                    root_lin_vel_cmd_body=cmd_body,
                    movement_mode=movement_mode,
                )

                foot_target = gait_out["foot_pos_target_rel"].clone()

                if movement_mode:
                    # Body-height proxy:
                    # Foot z is negative in body frame. Making feet more negative
                    # corresponds to a higher nominal body stance.
                    body_height_delta = float(getattr(self.cfg, "meta_body_height_delta", 0.04)) * theta[:, 2:3]
                    foot_target[:, 2, :] = foot_target[:, 2, :] - body_height_delta

                    # Swing-clearance proxy:
                    # Positive theta[3] increases clearance only on swing legs.
                    swing_mask = gait_out["swing_mask"].float()
                    clearance_extra = float(getattr(self.cfg, "meta_clearance_delta", 0.08)) * torch.relu(theta[:, 3:4])
                    foot_target[:, 2, :] = foot_target[:, 2, :] + clearance_extra * swing_mask

                    # Stance propulsion proxy:
                    # Positive TRACER vx means forward body motion.
                    # To push the body forward, stance feet should move slightly backward
                    # in the body frame. Keep this tiny; large values can destabilize contact.
                    stance_mask = 1.0 - swing_mask
                    stance_push_gain = float(getattr(self.cfg, "meta_stance_push_gain", 0.0))
                    if stance_push_gain != 0.0:
                        control_dt = float(getattr(self.gait_core.cfg, "control_dt", 0.02))
                        stance_dx = -vx_cmd.view(-1, 1) * control_dt * stance_push_gain
                        foot_target[:, 0, :] = foot_target[:, 0, :] + stance_dx * stance_mask

                    nominal_joint_target = self.a1_kin.ik_foot_targets(
                        self.robot.data.joint_pos.detach(),
                        foot_target,
                        num_iters=10,
                        damping=1.0e-3,
                        step_size=0.8,
                    )
                else:
                    nominal_joint_target = self.robot.data.default_joint_pos.detach().clone()

                if movement_mode:
                    # Safety V1:
                    # Apply IK target only to swing legs.
                    # Keep stance legs near the default support pose so the position drive
                    # does not pull supporting feet away and collapse the body.
                    swing_leg_mask = gait_out["swing_mask"].float()

                    swing_joint_mask = torch.zeros(self.num_envs, 12, device=self.device)
                    # IsaacLab joint order:
                    # [FL_hip, FR_hip, RL_hip, RR_hip,
                    #  FL_thigh, FR_thigh, RL_thigh, RR_thigh,
                    #  FL_calf, FR_calf, RL_calf, RR_calf]
                    swing_joint_mask[:, 0] = swing_leg_mask[:, 0]
                    swing_joint_mask[:, 1] = swing_leg_mask[:, 1]
                    swing_joint_mask[:, 2] = swing_leg_mask[:, 2]
                    swing_joint_mask[:, 3] = swing_leg_mask[:, 3]
                    swing_joint_mask[:, 4] = swing_leg_mask[:, 0]
                    swing_joint_mask[:, 5] = swing_leg_mask[:, 1]
                    swing_joint_mask[:, 6] = swing_leg_mask[:, 2]
                    swing_joint_mask[:, 7] = swing_leg_mask[:, 3]
                    swing_joint_mask[:, 8] = swing_leg_mask[:, 0]
                    swing_joint_mask[:, 9] = swing_leg_mask[:, 1]
                    swing_joint_mask[:, 10] = swing_leg_mask[:, 2]
                    swing_joint_mask[:, 11] = swing_leg_mask[:, 3]

                    stance_joint_default = self.robot.data.default_joint_pos.detach().clone()
                    stance_ik_blend = float(getattr(self.cfg, "meta_stance_ik_blend", 0.0))
                    stance_ik_blend = max(0.0, min(1.0, stance_ik_blend))
                    stance_joint_target = stance_joint_default + stance_ik_blend * (
                        nominal_joint_target - stance_joint_default
                    )
                    self._joint_pos_target = torch.where(
                        swing_joint_mask > 0.5,
                        nominal_joint_target,
                        stance_joint_target,
                    )
                else:
                    self._joint_pos_target = nominal_joint_target

                if bool(getattr(self.cfg, "meta_debug", False)) and int(self.common_step_counter) % 20 == 0:
                    print(
                        "[A1Adapter][meta_gait_theta] "
                        f"step={int(self.common_step_counter)} "
                        f"vx_mean={vx_cmd.mean().item():.3f} "
                        f"theta_mean={theta.mean(dim=0).detach().cpu().tolist()}",
                        flush=True,
                    )

                self._previous_action = self._actions.detach().clone()
                self._previous_adapter_action = adapter_action.detach().clone()
                return

            if bool(self.cfg.use_grf_torque):
                # GRF torque mode.
                #
                # If use_nominal_gait is False:
                #   all four legs are treated as stance.
                #
                # If use_nominal_gait is True:
                #   stance legs receive GRF torque,
                #   swing legs follow the nominal foot trajectory through IK.
                contact_mask = torch.ones(self.num_envs, 4, device=self.device)
                nominal_joint_target = self.robot.data.default_joint_pos.detach().clone()
                movement_mode = False

                if bool(self.cfg.use_nominal_gait):
                    foot_cur = self.a1_kin.fk_isaac(self.robot.data.joint_pos.detach())

                    cmd_body = torch.zeros(self.num_envs, 3, device=self.device)
                    cmd_body[:, 0] = float(self.cfg.gait_cmd_x)
                    cmd_body[:, 1] = float(self.cfg.gait_cmd_y)

                    movement_mode = bool(
                        torch.min(self.episode_length_buf).item() >= int(self.cfg.gait_warmup_steps)
                    )

                    gait_out = self.gait_core.step(
                        foot_pos_cur_rel=foot_cur,
                        root_lin_vel_body=self.robot.data.root_lin_vel_b.detach(),
                        root_lin_vel_cmd_body=cmd_body,
                        movement_mode=movement_mode,
                    )

                    if movement_mode:
                        # Prefer explicit plan_contacts if present.
                        # Fallback: stance = not swing.
                        if "plan_contacts" in gait_out:
                            contact_mask = gait_out["plan_contacts"].float()
                        else:
                            contact_mask = (~gait_out["swing_mask"]).float()

                        nominal_joint_target = self.a1_kin.ik_foot_targets(
                            self.robot.data.joint_pos.detach(),
                            gait_out["foot_pos_target_rel"],
                            num_iters=10,
                            damping=1.0e-3,
                            step_size=0.8,
                        )
                    else:
                        contact_mask = torch.ones(self.num_envs, 4, device=self.device)
                        nominal_joint_target = self.robot.data.default_joint_pos.detach().clone()

                foot_force = torch.zeros(self.num_envs, 3, 4, device=self.device)

                base_h = self.robot.data.root_pos_w[:, 2]
                base_vz = self.robot.data.root_lin_vel_w[:, 2]

                h_err = float(self.cfg.grf_height_target) - base_h
                az_cmd = (
                    9.81
                    + float(self.cfg.grf_height_kp) * h_err
                    - float(self.cfg.grf_height_kd) * base_vz
                )

                fz_total = float(self.cfg.grf_scale) * float(self.cfg.grf_mass) * az_cmd

                # Quasi-static vertical-force distribution.
                # Instead of uniform fz_total / stance_count, distribute stance-foot
                # vertical forces to match total Fz and roll/pitch correction moments.
                #
                # For vertical force fz_i at foot position r_i=(x_i, y_i, z_i):
                #   tau_roll  = tau_x =  y_i * fz_i
                #   tau_pitch = tau_y = -x_i * fz_i
                foot_pos_rel = self.a1_kin.fk_isaac(self.robot.data.joint_pos.detach())
                foot_x = foot_pos_rel[:, 0, :]
                foot_y = foot_pos_rel[:, 1, :]

                roll = obs_dict["roll"].squeeze(-1)
                pitch = obs_dict["pitch"].squeeze(-1)
                roll_rate = self.robot.data.root_ang_vel_b[:, 0]
                pitch_rate = self.robot.data.root_ang_vel_b[:, 1]

                grf_roll_kp = float(getattr(self.cfg, "grf_roll_kp", 8.0))
                grf_roll_kd = float(getattr(self.cfg, "grf_roll_kd", 1.0))
                grf_pitch_kp = float(getattr(self.cfg, "grf_pitch_kp", 8.0))
                grf_pitch_kd = float(getattr(self.cfg, "grf_pitch_kd", 1.0))
                grf_wrench_damping = float(getattr(self.cfg, "grf_wrench_damping", 1.0e-3))

                tau_roll_des = -grf_roll_kp * roll - grf_roll_kd * roll_rate
                tau_pitch_des = -grf_pitch_kp * pitch - grf_pitch_kd * pitch_rate

                A = torch.zeros(self.num_envs, 3, 4, device=self.device)
                A[:, 0, :] = contact_mask
                A[:, 1, :] = foot_y * contact_mask
                A[:, 2, :] = -foot_x * contact_mask

                b = torch.stack([fz_total, tau_roll_des, tau_pitch_des], dim=1)

                eye3 = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
                lhs = torch.bmm(A, A.transpose(1, 2)) + grf_wrench_damping * eye3
                y = torch.linalg.solve(lhs, b.unsqueeze(-1))
                fz = torch.bmm(A.transpose(1, 2), y).squeeze(-1)

                fz = torch.clamp(
                    fz,
                    float(self.cfg.grf_fz_min),
                    float(self.cfg.grf_fz_max),
                )
                fz = fz * contact_mask

                # Horizontal stance force for coarse velocity tracking.
                # This is still not full MPC, but it gives the body a commanded
                # fore-aft/lateral push through stance legs.
                grf_vx_kp = float(getattr(self.cfg, "grf_vx_kp", 8.0))
                grf_vy_kp = float(getattr(self.cfg, "grf_vy_kp", 4.0))
                grf_fx_max = float(getattr(self.cfg, "grf_fx_max", 20.0))
                grf_fy_max = float(getattr(self.cfg, "grf_fy_max", 15.0))

                if bool(self.cfg.use_nominal_gait) and (not movement_mode):
                    # During warmup/settling, do not apply horizontal velocity tracking.
                    # Otherwise the robot drifts before the gait actually starts.
                    vx_cmd = 0.0
                    vy_cmd = 0.0
                else:
                    vx_cmd = float(self.cfg.gait_cmd_x)
                    vy_cmd = float(self.cfg.gait_cmd_y)

                vx = self.robot.data.root_lin_vel_b[:, 0]
                vy = self.robot.data.root_lin_vel_b[:, 1]

                # Note: because the final torque path uses grf_sign=-1.0,
                # the commanded horizontal foot force is opposite to the desired
                # body acceleration direction.
                #
                # Fx: forward velocity tracking.
                # Fy: lateral position hold + lateral velocity damping.
                root_xy_now = self.robot.data.root_pos_w[:, :2]

                if (not hasattr(self, "_grf_xy_ref")) or (self._grf_xy_ref.shape[0] != self.num_envs):
                    self._grf_xy_ref = root_xy_now.detach().clone()

                # During warmup, keep updating the lateral reference so that
                # the controller holds the settled pose, not the spawn pose.
                if bool(self.cfg.use_nominal_gait) and (not movement_mode):
                    self._grf_xy_ref = root_xy_now.detach().clone()

                y_ref = self._grf_xy_ref[:, 1]
                y_now = root_xy_now[:, 1]
                y_err = y_ref - y_now

                grf_y_pos_kp = float(getattr(self.cfg, "grf_y_pos_kp", 2.0))

                fx_total = -float(self.cfg.grf_mass) * grf_vx_kp * (vx_cmd - vx)

                # Keep Fy with the original lateral sign.
                # When the body drifts to +y, y_err = y_ref - y_now becomes negative,
                # so this produces a corrective -Fy command.
                fy_total = float(self.cfg.grf_mass) * (grf_y_pos_kp * y_err + grf_vy_kp * (vy_cmd - vy))

                fx_total = torch.clamp(fx_total, -grf_fx_max, grf_fx_max)
                fy_total = torch.clamp(fy_total, -grf_fy_max, grf_fy_max)

                # Distribute horizontal forces over stance feet while also damping yaw.
                #
                # Variables:
                #   u = [fx_FL, fx_FR, fx_RL, fx_RR, fy_FL, fy_FR, fy_RL, fy_RR]
                #
                # Constraints:
                #   sum fx_i = Fx_total
                #   sum fy_i = Fy_total
                #   sum (x_i * fy_i - y_i * fx_i) = tau_yaw_des
                #
                # This prevents the horizontal push from creating large unintended
                # yaw/lateral drift during asymmetric crawl support.
                yaw = obs_dict["yaw"].squeeze(-1)
                yaw_rate = self.robot.data.root_ang_vel_b[:, 2]

                grf_yaw_kp = float(getattr(self.cfg, "grf_yaw_kp", 2.0))
                grf_yaw_kd = float(getattr(self.cfg, "grf_yaw_kd", 0.5))
                grf_xy_wrench_damping = float(getattr(self.cfg, "grf_xy_wrench_damping", 1.0e-3))

                tau_yaw_des = -grf_yaw_kp * yaw - grf_yaw_kd * yaw_rate

                Axy = torch.zeros(self.num_envs, 3, 8, device=self.device)

                # total Fx
                Axy[:, 0, 0:4] = contact_mask

                # total Fy
                Axy[:, 1, 4:8] = contact_mask

                # yaw moment: tau_z = x * fy - y * fx
                Axy[:, 2, 0:4] = -foot_y * contact_mask
                Axy[:, 2, 4:8] = foot_x * contact_mask

                bxy = torch.stack([fx_total, fy_total, tau_yaw_des], dim=1)

                eye_xy = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
                lhs_xy = torch.bmm(Axy, Axy.transpose(1, 2)) + grf_xy_wrench_damping * eye_xy
                yxy = torch.linalg.solve(lhs_xy, bxy.unsqueeze(-1))
                uxy = torch.bmm(Axy.transpose(1, 2), yxy).squeeze(-1)

                fx_each = torch.clamp(uxy[:, 0:4], -grf_fx_max, grf_fx_max) * contact_mask
                fy_each = torch.clamp(uxy[:, 4:8], -grf_fy_max, grf_fy_max) * contact_mask

                foot_force[:, 0, :] = fx_each
                foot_force[:, 1, :] = fy_each
                foot_force[:, 2, :] = fz

                tau = self.a1_kin.foot_forces_to_joint_torques(
                    self.robot.data.joint_pos.detach(),
                    foot_force,
                )
                tau = float(self.cfg.grf_sign) * tau
                tau = tau - float(self.cfg.grf_joint_damping) * self.robot.data.joint_vel.detach()
                tau = torch.clamp(
                    tau,
                    -float(self.cfg.grf_tau_limit),
                    float(self.cfg.grf_tau_limit),
                )

                self._joint_effort_target = tau

                if bool(self.cfg.use_nominal_gait) and movement_mode:
                    # Apply position tracking only to swing legs.
                    # For stance legs, use current joint position so the position drive
                    # does not fight against the GRF torque support.
                    swing_leg_mask = 1.0 - contact_mask

                    swing_joint_mask = torch.zeros(self.num_envs, 12, device=self.device)
                    # IsaacLab joint order:
                    # [FL_hip, FR_hip, RL_hip, RR_hip,
                    #  FL_thigh, FR_thigh, RL_thigh, RR_thigh,
                    #  FL_calf, FR_calf, RL_calf, RR_calf]
                    swing_joint_mask[:, 0] = swing_leg_mask[:, 0]
                    swing_joint_mask[:, 1] = swing_leg_mask[:, 1]
                    swing_joint_mask[:, 2] = swing_leg_mask[:, 2]
                    swing_joint_mask[:, 3] = swing_leg_mask[:, 3]
                    swing_joint_mask[:, 4] = swing_leg_mask[:, 0]
                    swing_joint_mask[:, 5] = swing_leg_mask[:, 1]
                    swing_joint_mask[:, 6] = swing_leg_mask[:, 2]
                    swing_joint_mask[:, 7] = swing_leg_mask[:, 3]
                    swing_joint_mask[:, 8] = swing_leg_mask[:, 0]
                    swing_joint_mask[:, 9] = swing_leg_mask[:, 1]
                    swing_joint_mask[:, 10] = swing_leg_mask[:, 2]
                    swing_joint_mask[:, 11] = swing_leg_mask[:, 3]

                    stance_joint_target = self.robot.data.default_joint_pos.detach().clone()
                    pos_target = torch.where(
                        swing_joint_mask > 0.5,
                        nominal_joint_target,
                        stance_joint_target,
                    )
                else:
                    pos_target = nominal_joint_target

                self._joint_pos_target = pos_target + self._residual_scale * self._actions

            elif self._hold_default_pose:
                self._joint_effort_target = torch.zeros_like(self.robot.data.default_joint_pos)
                self._joint_pos_target = self.robot.data.default_joint_pos.detach().clone()

            elif bool(self.cfg.use_nominal_gait):
                self._joint_effort_target = torch.zeros_like(self.robot.data.default_joint_pos)

                foot_cur = self.a1_kin.fk_isaac(self.robot.data.joint_pos.detach())

                cmd_body = torch.zeros(self.num_envs, 3, device=self.device)
                cmd_body[:, 0] = float(self.cfg.gait_cmd_x)
                cmd_body[:, 1] = float(self.cfg.gait_cmd_y)

                movement_mode = bool(
                    torch.min(self.episode_length_buf).item() >= int(self.cfg.gait_warmup_steps)
                )

                gait_out = self.gait_core.step(
                    foot_pos_cur_rel=foot_cur,
                    root_lin_vel_body=self.robot.data.root_lin_vel_b.detach(),
                    root_lin_vel_cmd_body=cmd_body,
                    movement_mode=movement_mode,
                )

                nominal_joint_target = self.a1_kin.ik_foot_targets(
                    self.robot.data.joint_pos.detach(),
                    gait_out["foot_pos_target_rel"],
                    num_iters=10,
                    damping=1.0e-3,
                    step_size=0.8,
                )

                if not movement_mode:
                    nominal_joint_target = self.robot.data.default_joint_pos.detach().clone()

                self._joint_pos_target = nominal_joint_target + self._residual_scale * self._actions

            else:
                self._joint_effort_target = torch.zeros_like(self.robot.data.default_joint_pos)
                self._joint_pos_target = out.joint_pos_target

            self._previous_action = self._actions.detach().clone()
            self._previous_adapter_action = adapter_action.detach().clone()

        def _apply_action(self):
            if self._joint_pos_target is None:
                self._joint_pos_target = self.robot.data.default_joint_pos.detach().clone()

            if self._joint_effort_target is None:
                self._joint_effort_target = torch.zeros_like(self.robot.data.default_joint_pos)

            if bool(getattr(self.cfg, "use_grf_torque", False)) or bool(getattr(self.cfg, "use_external_lowlevel", False)):
                self.robot.set_joint_position_target(self._joint_pos_target)
                self.robot.set_joint_effort_target(self._joint_effort_target)
            else:
                self.robot.set_joint_position_target(self._joint_pos_target)

        def _get_observations(self):
            if self._policy_obs is None:
                self._policy_obs = torch.zeros(
                    self.num_envs,
                    self.cfg.observation_space,
                    device=self.device,
                )

            return {"policy": self._policy_obs}

        def _get_rewards(self):
            action_type = str(getattr(self.cfg, "action_type", "joint_position_residual")).lower()

            if action_type == "meta_gait_theta":
                # PPO-facing reward for the structured TRACER meta-gait action.
                #
                # Coordinate convention:
                # - We reward actual root/body forward velocity in world x.
                # - The low-level gait command sign may be flipped internally
                #   through meta_gait_x_sign, but reward should measure real motion.
                root_lin_vel_w = self.robot.data.root_lin_vel_w
                root_lin_vel_b = self.robot.data.root_lin_vel_b
                root_ang_vel_b = self.robot.data.root_ang_vel_b
                root_height = self.robot.data.root_pos_w[:, 2]

                forward_vel = root_lin_vel_w[:, 0]
                lateral_vel = root_lin_vel_w[:, 1]
                yaw_rate = root_ang_vel_b[:, 2]
                ang_vel_xy = torch.linalg.norm(root_ang_vel_b[:, :2], dim=-1)

                height_err = root_height - float(getattr(self.cfg, "meta_target_height", 0.285))
                low_height = torch.relu(float(getattr(self.cfg, "meta_safe_min_height", 0.255)) - root_height)

                joint_vel = self.robot.data.joint_vel
                joint_vel_pen = torch.mean(torch.square(joint_vel), dim=-1)

                if hasattr(self.robot.data, "applied_torque") and self.robot.data.applied_torque is not None:
                    torque = self.robot.data.applied_torque
                    torque_pen = torch.mean(torch.square(torque), dim=-1)
                else:
                    torque_pen = torch.zeros(self.num_envs, device=self.device)

                if self._actions is None:
                    action_pen = torch.zeros(self.num_envs, device=self.device)
                else:
                    action_pen = torch.mean(torch.square(self._actions), dim=-1)

                if self._previous_action is None or self._actions is None:
                    action_rate_pen = torch.zeros(self.num_envs, device=self.device)
                else:
                    # If action dimensions differ due to old buffers, skip safely.
                    if self._previous_action.shape == self._actions.shape:
                        action_rate_pen = torch.mean(torch.square(self._actions - self._previous_action), dim=-1)
                    else:
                        action_rate_pen = torch.zeros(self.num_envs, device=self.device)

                fall = root_height < 0.18

                # Legacy V0 reward path is the default.
                #
                # Important: the flat-ground V0 learner is our stable regression baseline.
                # The beta-weighted reward scaffold is only enabled explicitly through
                # cfg.meta_use_beta_reward / TRACER_META_USE_BETA_REWARD or TRACER_META_BETA.
                use_beta_reward = bool(getattr(self.cfg, "meta_use_beta_reward", False))
                
                if not use_beta_reward:
                    reward = torch.zeros(self.num_envs, device=self.device)
                    reward = reward + float(getattr(self.cfg, "meta_reward_forward_vel", 4.0)) * forward_vel
                    reward = reward + float(getattr(self.cfg, "meta_reward_forward_progress", 2.0)) * torch.relu(forward_vel)
                
                    reward = reward - float(getattr(self.cfg, "meta_penalty_lateral_vel", 0.5)) * torch.square(lateral_vel)
                    reward = reward - float(getattr(self.cfg, "meta_penalty_yaw_rate", 0.15)) * torch.square(yaw_rate)
                    reward = reward - float(getattr(self.cfg, "meta_penalty_ang_vel", 0.05)) * torch.square(ang_vel_xy)
                    reward = reward - float(getattr(self.cfg, "meta_penalty_height", 4.0)) * torch.square(height_err)
                    reward = reward - float(getattr(self.cfg, "meta_penalty_low_height", 8.0)) * torch.square(low_height)
                    reward = reward - float(getattr(self.cfg, "meta_penalty_joint_vel", 0.002)) * joint_vel_pen
                    reward = reward - float(getattr(self.cfg, "meta_penalty_torque", 0.0005)) * torque_pen
                    reward = reward - float(getattr(self.cfg, "meta_penalty_action", 0.02)) * action_pen
                    reward = reward - float(getattr(self.cfg, "meta_penalty_action_rate", 0.02)) * action_rate_pen
                    reward = reward - float(getattr(self.cfg, "meta_fall_penalty", 10.0)) * fall.float()
                
                    self._last_meta_reward_terms = {
                        "reward_mode": "legacy_v0",
                    }
                    return reward
                
                # Beta-weighted TRACER reward scaffold.
                velocity_reward = (
                    float(getattr(self.cfg, "meta_reward_forward_vel", 4.0)) * forward_vel
                    + float(getattr(self.cfg, "meta_reward_forward_progress", 2.0)) * torch.relu(forward_vel)
                )
                
                stability_penalty = (
                    float(getattr(self.cfg, "meta_penalty_lateral_vel", 0.5)) * torch.square(lateral_vel)
                    + float(getattr(self.cfg, "meta_penalty_yaw_rate", 0.15)) * torch.square(yaw_rate)
                    + float(getattr(self.cfg, "meta_penalty_ang_vel", 0.05)) * torch.square(ang_vel_xy)
                )
                
                posture_penalty = (
                    float(getattr(self.cfg, "meta_penalty_height", 4.0)) * torch.square(height_err)
                    + float(getattr(self.cfg, "meta_penalty_low_height", 8.0)) * torch.square(low_height)
                )
                
                energy_penalty = (
                    float(getattr(self.cfg, "meta_penalty_joint_vel", 0.002)) * joint_vel_pen
                    + float(getattr(self.cfg, "meta_penalty_torque", 0.0005)) * torque_pen
                )
                
                action_regularization = (
                    float(getattr(self.cfg, "meta_penalty_action", 0.02)) * action_pen
                    + float(getattr(self.cfg, "meta_penalty_action_rate", 0.02)) * action_rate_pen
                )
                
                theta = getattr(self, "_last_meta_theta", None)
                if theta is not None and theta.shape[-1] > 3:
                    clearance_proxy = torch.relu(theta[:, 3])
                else:
                    clearance_proxy = torch.zeros(self.num_envs, device=self.device)
                
                clearance_reward = float(getattr(self.cfg, "meta_reward_clearance", 0.0)) * clearance_proxy
                
                beta_v = float(getattr(self.cfg, "meta_beta_velocity", 1.0))
                beta_s = float(getattr(self.cfg, "meta_beta_stability", 1.0))
                beta_e = float(getattr(self.cfg, "meta_beta_energy", 1.0))
                beta_c = float(getattr(self.cfg, "meta_beta_clearance", 0.0))
                
                fall_penalty = float(getattr(self.cfg, "meta_fall_penalty", 10.0)) * fall.float()
                
                reward = torch.zeros(self.num_envs, device=self.device)
                reward = reward + beta_v * velocity_reward
                reward = reward - beta_s * (stability_penalty + posture_penalty)
                reward = reward - beta_e * energy_penalty
                reward = reward + beta_c * clearance_reward
                reward = reward - action_regularization
                reward = reward - fall_penalty
                
                self._last_meta_reward_terms = {
                    "reward_mode": "beta_weighted",
                    "velocity_reward": velocity_reward.detach(),
                    "stability_penalty": stability_penalty.detach(),
                    "posture_penalty": posture_penalty.detach(),
                    "energy_penalty": energy_penalty.detach(),
                    "action_regularization": action_regularization.detach(),
                    "clearance_reward": clearance_reward.detach(),
                    "fall_penalty": fall_penalty.detach(),
                    "beta_velocity": beta_v,
                    "beta_stability": beta_s,
                    "beta_energy": beta_e,
                    "beta_clearance": beta_c,
                }
                
                return reward

            if self._reward is None:
                return torch.zeros(self.num_envs, device=self.device)

            return self._reward

        def _get_dones(self):
            if self._terminated is None:
                terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            else:
                terminated = self._terminated

            root_pos = self.robot.data.root_pos_w
            root_height = root_pos[:, 2]

            action_type = str(getattr(self.cfg, "action_type", "joint_position_residual")).lower()
            if action_type == "meta_gait_theta":
                height_threshold = float(getattr(self.cfg, "meta_done_min_height", 0.22))
            else:
                height_threshold = 0.18

            height_fail = root_height < height_threshold
            terminated = torch.logical_or(terminated, height_fail)

            if action_type == "meta_gait_theta":
                # Keep early PPO rollouts inside a conservative stable basin.
                env_origins = self.scene.env_origins
                rel_x = root_pos[:, 0] - env_origins[:, 0]
                rel_y = root_pos[:, 1] - env_origins[:, 1]

                lateral_fail = torch.abs(rel_y) > float(getattr(self.cfg, "meta_done_lateral_dist", 0.35))
                backward_fail = rel_x < -float(getattr(self.cfg, "meta_done_backward_dist", 0.25))
                terminated = torch.logical_or(terminated, lateral_fail)
                terminated = torch.logical_or(terminated, backward_fail)

            if self._truncated is None:
                truncated = self.episode_length_buf >= self.max_episode_length - 1
            else:
                truncated = torch.logical_or(
                    self._truncated,
                    self.episode_length_buf >= self.max_episode_length - 1,
                )

            return terminated, truncated

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)

            self._init_tracer_if_needed()

            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

            # Explicitly reset the robot to IsaacLab's default standing state.
            root_state = self.robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += self.scene.env_origins[env_ids]

            # Use FK-consistent reset height instead of asset default height.
            # This avoids dropping from an overly high spawn pose before the
            # controller/gait branch even starts.
            root_state[:, 2] = self.scene.env_origins[env_ids, 2] + float(self.cfg.reset_root_height)

            joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
            joint_vel = torch.zeros_like(joint_pos)

            self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
            self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

            self.tracer_adapter.reset(env_ids)

            if self.gait_core is not None and self.a1_kin is not None:
                default_foot = self.a1_kin.fk_isaac(self.robot.data.default_joint_pos.detach())
                self.gait_core.set_default_foot_pos(default_foot)
                self.gait_core.reset(env_ids)

            if self._previous_action is None:
                self._previous_action = torch.zeros(
                    self.num_envs,
                    self.cfg.action_space,
                    device=self.device,
                )
            self._previous_action[env_ids] = 0.0
            if hasattr(self, "_meta_theta_smooth") and self._meta_theta_smooth is not None:
                self._meta_theta_smooth[env_ids] = 0.0

            if self._policy_obs is None:
                self._policy_obs = torch.zeros(
                    self.num_envs,
                    self.cfg.observation_space,
                    device=self.device,
                )
            self._policy_obs[env_ids] = 0.0

            self._reward = None
            self._terminated = None
            self._truncated = None
            self._joint_pos_target = self.robot.data.default_joint_pos.detach().clone()

    return TracerA1AdapterEnv, TracerA1AdapterEnvCfg
