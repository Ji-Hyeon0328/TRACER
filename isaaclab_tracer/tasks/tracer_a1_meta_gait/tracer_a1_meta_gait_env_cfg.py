from __future__ import annotations

from isaaclab_tracer.envs.tracer_a1_adapter_env import make_tracer_a1_adapter_env_class


TracerA1MetaGaitEnv, _TracerA1AdapterEnvCfg = make_tracer_a1_adapter_env_class()


class TracerA1MetaGaitEnvCfg(_TracerA1AdapterEnvCfg):
    """TRACER A1 meta-gait task config for Isaac Lab RL runners.

    V0 keeps the safe curriculum:
      - action_type = meta_gait_theta
      - action dim = 6
      - internally trains only theta[0] through meta_train_vx_only
    """

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()

        # RL-facing TRACER meta-gait action.
        self.action_type = "meta_gait_theta"

        # V0 exposes only theta[0] to rsl_rl.
        # The env pads this 1D action to full meta_theta_dim internally.
        self.action_space = 1

        # Keep the stable internal low-level gait branch.
        self.ignore_adapter_done = True
        self.meta_debug = False

        self.use_external_lowlevel = False
        self.external_lowlevel_timeout_s = 0.20
        self.external_lowlevel_env_index = 0
        self.use_grf_torque = False
        self.use_nominal_gait = True
        self.hold_default_pose = False
        self.residual_scale = 0.0

        # V0 curriculum: only theta[0] is active.
        self.meta_train_vx_only = True
        self.meta_theta_smoothing_alpha = 0.90

        # Conservative reset/done safety.
        self.meta_done_min_height = 0.22
        self.meta_done_lateral_dist = 0.35
        self.meta_done_backward_dist = 0.25

        # Keep the same stable gait prior.
        self.reset_root_height = 0.385
        self.gait_pattern = "trot"
        self.gait_warmup_steps = 80
        self.gait_counter_speed = 2.0
        self.gait_clearance2 = 0.03
        self.gait_foot_delta_x_limit = 0.05
        self.gait_foot_delta_y_limit = 0.03
        self.adapter_actuator_stiffness = 60.0
        self.adapter_actuator_damping = 2.0
        self.meta_base_vx = 0.10
        self.meta_gait_x_sign = -1.0
        self.meta_stance_push_gain = 1.0
        self.meta_stance_ik_blend = 0.6
