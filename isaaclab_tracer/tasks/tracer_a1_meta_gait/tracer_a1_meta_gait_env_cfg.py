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

        # Optional hand-coded terrain context appended to observation.
        #
        # Default off:
        #   observation_space = 58
        #
        # If TRACER_INCLUDE_TERRAIN_CONTEXT=1:
        #   observation_space = 58 + 5
        #   obs += [static_friction, dynamic_friction, is_rough, is_slippery, is_soft]
        include_ctx_raw = os.environ.get("TRACER_INCLUDE_TERRAIN_CONTEXT", "0").strip().lower()
        self.include_terrain_context_obs = include_ctx_raw in ("1", "true", "yes", "on")
        self.terrain_context_dim = 5
        if self.include_terrain_context_obs:
            self.observation_space = int(self.observation_space) + int(self.terrain_context_dim)

        # Physical terrain/material preset.
        #
        # This affects the Isaac ground-plane material. It is intentionally
        # separate from TRACER_TERRAIN_BETA_PRESET:
        #   TRACER_TERRAIN_PRESET      -> physical/material condition
        #   TRACER_TERRAIN_BETA_PRESET -> objective/reward weighting
        terrain_raw = os.environ.get("TRACER_TERRAIN_PRESET", "").strip().lower()
        if terrain_raw:
            if terrain_raw in ("flat", "solid", "solid_even"):
                self.terrain_preset = "solid"
                self.terrain_static_friction = 1.0
                self.terrain_dynamic_friction = 1.0
                self.terrain_restitution = 0.0
            elif terrain_raw in ("rough",):
                # Material-only rough scaffold.
                self.terrain_preset = "rough"
                self.terrain_static_friction = 1.2
                self.terrain_dynamic_friction = 1.0
                self.terrain_restitution = 0.0
            elif terrain_raw in ("rough_bumps", "solid_rough_bumps"):
                # Geometry roughness scaffold: low cuboid bump bars.
                # Keep material flat-like; roughness should come from geometry, not friction.
                self.terrain_preset = "rough_bumps"
                self.terrain_static_friction = 1.0
                self.terrain_dynamic_friction = 1.0
                self.terrain_restitution = 0.0
            elif terrain_raw in ("slippery", "icy", "icy_slippery", "ice"):
                # Approximation of icy/slippery contact: low static/dynamic friction.
                self.terrain_preset = "icy_slippery"
                self.terrain_static_friction = 0.25
                self.terrain_dynamic_friction = 0.20
                self.terrain_restitution = 0.0
            elif terrain_raw in ("mud", "mud_slippery", "muddy"):
                # Approximation of muddy slip: moderate-low friction.
                # True sinkage/deformability will be added later as a separate proxy.
                self.terrain_preset = "mud_slippery"
                self.terrain_static_friction = 0.45
                self.terrain_dynamic_friction = 0.30
                self.terrain_restitution = 0.0
            elif terrain_raw in ("soft", "sponge", "sponge_like", "sponge-like"):
                # Approximation of sponge-like contact.
                # True compliance will be added later as a separate proxy.
                self.terrain_preset = "sponge_like"
                self.terrain_static_friction = 0.70
                self.terrain_dynamic_friction = 0.55
                self.terrain_restitution = 0.0
            else:
                raise ValueError(
                    f"Unknown TRACER_TERRAIN_PRESET={terrain_raw!r}. "
                    "Expected one of: flat/solid, rough, rough_bumps, "
                    "slippery/icy_slippery, mud_slippery, soft/sponge_like."
                )

        # Optional direct friction override:
        #   TRACER_TERRAIN_FRICTION=0.8
        #   TRACER_TERRAIN_FRICTION=0.8,0.6
        friction_raw = os.environ.get("TRACER_TERRAIN_FRICTION", "").strip()
        if friction_raw:
            friction_vals = [float(x.strip()) for x in friction_raw.split(",") if x.strip() != ""]
            if len(friction_vals) == 1:
                self.terrain_static_friction = friction_vals[0]
                self.terrain_dynamic_friction = friction_vals[0]
            elif len(friction_vals) == 2:
                self.terrain_static_friction = friction_vals[0]
                self.terrain_dynamic_friction = friction_vals[1]
            else:
                raise ValueError(
                    "TRACER_TERRAIN_FRICTION must have one or two values: "
                    "static[,dynamic]"
                )

        # Optional rough bump geometry overrides.
        if os.environ.get("TRACER_ROUGH_BUMP_COUNT", "").strip():
            self.terrain_rough_bump_count = int(os.environ["TRACER_ROUGH_BUMP_COUNT"])
        if os.environ.get("TRACER_ROUGH_BUMP_HEIGHT", "").strip():
            self.terrain_rough_bump_height = float(os.environ["TRACER_ROUGH_BUMP_HEIGHT"])
        if os.environ.get("TRACER_ROUGH_BUMP_LENGTH", "").strip():
            self.terrain_rough_bump_length = float(os.environ["TRACER_ROUGH_BUMP_LENGTH"])
        if os.environ.get("TRACER_ROUGH_BUMP_WIDTH", "").strip():
            self.terrain_rough_bump_width = float(os.environ["TRACER_ROUGH_BUMP_WIDTH"])
        if os.environ.get("TRACER_ROUGH_BUMP_START_X", "").strip():
            self.terrain_rough_bump_start_x = float(os.environ["TRACER_ROUGH_BUMP_START_X"])
        if os.environ.get("TRACER_ROUGH_BUMP_SPACING_X", "").strip():
            self.terrain_rough_bump_spacing_x = float(os.environ["TRACER_ROUGH_BUMP_SPACING_X"])
        if os.environ.get("TRACER_ROUGH_BUMP_Y", "").strip():
            self.terrain_rough_bump_y = float(os.environ["TRACER_ROUGH_BUMP_Y"])

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

        # Terrain-aware beta presets.
        #
        # These are hand-coded placeholders for the future Objective Selector.
        # Later, the Objective Selector should predict beta from terrain/context.
        #
        # Priority interpretation:
        #   beta_velocity  : forward progress priority
        #   beta_stability : posture / slip-safe / low-risk priority
        #   beta_energy    : energy-efficiency priority
        #   beta_clearance : clearance/style priority
        preset_raw = os.environ.get("TRACER_TERRAIN_BETA_PRESET", "").strip().lower()
        if preset_raw and not beta_raw.strip():
            if preset_raw == "flat":
                beta_raw = "1.0,1.0,1.0,0.0"
                self.meta_reward_clearance = 0.0
            elif preset_raw == "rough":
                beta_raw = "0.8,1.5,1.0,1.0"
                self.meta_reward_clearance = 0.05
            elif preset_raw == "rough_bumps":
                # TRACER-consistent setting: beta changes objective priorities,
                # but theta/clearance is not directly rewarded.
                beta_raw = "0.7,1.6,1.0,0.0"
                self.meta_reward_clearance = 0.0
            elif preset_raw == "slippery":
                beta_raw = "0.5,2.0,0.8,0.2"
                self.meta_reward_clearance = 0.0
            elif preset_raw == "soft":
                beta_raw = "0.6,1.7,1.2,0.2"
                self.meta_reward_clearance = 0.0
            else:
                raise ValueError(
                    f"Unknown TRACER_TERRAIN_BETA_PRESET={preset_raw!r}. "
                    "Expected one of: flat, rough, rough_bumps, slippery, soft."
                )

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

        # Reward-mode gate.
        # auto:
        #   default V0 uses the legacy fixed reward;
        #   providing TRACER_META_BETA or TRACER_TERRAIN_BETA_PRESET
        #   automatically enables beta-weighted reward.
        use_beta_raw = os.environ.get("TRACER_META_USE_BETA_REWARD", "auto").strip().lower()
        if use_beta_raw == "auto":
            self.meta_use_beta_reward = bool(beta_raw.strip())
        else:
            self.meta_use_beta_reward = use_beta_raw in ("1", "true", "yes", "on")

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
        # Gait direction is a traversal primitive, not a terrain material.
        #
        # Current convention:
        #   forward  : meta_gait_x_sign=-1 makes the robot move +world-x.
        #   backward : flip the low-level command sign and reward -world-x progress.
        gait_direction_raw = os.environ.get("TRACER_GAIT_DIRECTION", "forward").strip().lower()
        if gait_direction_raw in ("forward", "fwd"):
            self.gait_direction = "forward"
            self.meta_gait_x_sign = -1.0
            self.meta_progress_sign = 1.0
        elif gait_direction_raw in ("backward", "back", "bwd"):
            self.gait_direction = "backward"
            self.meta_gait_x_sign = 1.0
            self.meta_progress_sign = -1.0
        else:
            raise ValueError(
                f"Unknown TRACER_GAIT_DIRECTION={gait_direction_raw!r}. "
                "Expected one of: forward, backward."
            )
        self.meta_stance_push_gain = 1.0
        self.meta_stance_ik_blend = 0.6
