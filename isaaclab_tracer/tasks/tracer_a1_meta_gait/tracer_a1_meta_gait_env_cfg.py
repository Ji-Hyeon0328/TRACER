from __future__ import annotations

import os

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

        # Curriculum-controlled active theta dimensions.
        # Default V0: "0" -> action[0] maps to theta[0].
        # V1 examples:
        #   TRACER_META_ACTIVE_THETA=0,2 -> theta[0] + theta[2]
        #   TRACER_META_ACTIVE_THETA=0,3 -> theta[0] + theta[3]
        # Curriculum-controlled active theta dimensions.
        # Default V0: "0" -> action[0] maps to theta[0].
        # Example V1: TRACER_META_ACTIVE_THETA=0,3
        active_raw = os.environ.get("TRACER_META_ACTIVE_THETA", "0")
        self.meta_active_theta_indices = [
            int(x.strip()) for x in active_raw.split(",") if x.strip() != ""
        ]
        
        # Optional per-active-dimension scale.
        # Example: TRACER_META_ACTIVE_THETA_SCALE=1.0,0.25
        scale_raw = os.environ.get("TRACER_META_ACTIVE_THETA_SCALE", "")
        if scale_raw.strip():
            self.meta_active_theta_scales = [
                float(x.strip()) for x in scale_raw.split(",") if x.strip() != ""
            ]
        else:
            self.meta_active_theta_scales = [
                1.0 for _ in self.meta_active_theta_indices
            ]
        
        self.action_space = len(self.meta_active_theta_indices)

        # Optional hand-coded beta override for terrain-aware reward experiments.
        # Format:
        #   TRACER_META_BETA=velocity,stability,energy
        #   TRACER_META_BETA=velocity,stability,energy,clearance
        #
        # Examples:
        #   flat:      1.0,1.0,1.0,0.0
        #   rough:     0.8,1.5,1.0,1.0
        #   slippery:  0.5,2.0,0.8,0.5
        beta_raw = os.environ.get("TRACER_META_BETA", "")
        if beta_raw.strip():
            beta_vals = [float(x.strip()) for x in beta_raw.split(",") if x.strip() != ""]
            if len(beta_vals) not in (3, 4):
                raise ValueError(
                    "TRACER_META_BETA must have 3 or 4 values: "
                    "velocity,stability,energy[,clearance]"
                )
            self.meta_beta_velocity = beta_vals[0]
            self.meta_beta_stability = beta_vals[1]
            self.meta_beta_energy = beta_vals[2]
            self.meta_beta_clearance = beta_vals[3] if len(beta_vals) == 4 else 0.0

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

        # V0 curriculum keeps the old vx-only safety mask.
        # For multi-theta curricula such as active=[0,3], disable vx-only masking
        # so the additional active theta dimension can actually affect the decoder.
        vx_only_raw = os.environ.get("TRACER_META_TRAIN_VX_ONLY", "auto").strip().lower()
        if vx_only_raw == "auto":
            self.meta_train_vx_only = self.meta_active_theta_indices == [0]
        else:
            self.meta_train_vx_only = vx_only_raw in ("1", "true", "yes", "on")

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
