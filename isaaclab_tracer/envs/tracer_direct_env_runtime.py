from typing import Dict, Any


def load_direct_env_symbols():
    """
    Load Isaac Lab DirectRLEnv symbols.

    This must be called after Isaac Sim runtime is launched through AppLauncher.
    Importing isaaclab.envs before AppLauncher may fail because pxr/omni modules
    are not available yet.
    """
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    return DirectRLEnv, DirectRLEnvCfg


class TracerDirectEnvLogicMixin:
    """
    TRACER-side logic mixin for a future Isaac Lab DirectRLEnv.

    This class intentionally does not depend on Isaac Lab imports.
    It only defines the expected TRACER integration points.
    """

    def _build_tracer_modules(self):
        """
        Future M30/M31 target:

            TracerTaskSpec
            TracerHighLevelLoop
            TracerStepAdapter

        will be initialized here after the real Isaac Lab env has num_envs,
        device, robot default joint positions, and scene objects.
        """
        raise NotImplementedError("TRACER module construction is scheduled for M30/M31.")

    def _extract_tracer_obs_dict(self) -> Dict[str, Any]:
        """
        Future M30/M31 target:

        Extract Isaac Lab robot tensors and convert them into the obs dict used by:

            TracerHighLevelLoop
            TracerStepAdapter

        Required keys:

            base_xy
            base_yaw
            base_lin_vel_body
            base_ang_vel_body
            base_yaw_rate
            projected_gravity
            height_scan
            base_height
            roll
            pitch
            joint_pos
            joint_vel
            previous_action
        """
        raise NotImplementedError("Isaac robot tensor extraction is scheduled for M30/M31.")

    def _compute_tracer_policy_obs(self):
        """
        Future target:

            obs_dict
            -> TracerStepAdapter
            -> policy_obs

        This becomes the core of DirectRLEnv._get_observations().
        """
        raise NotImplementedError("TRACER policy observation wiring is scheduled for M30/M31.")


def make_tracer_direct_env_class():
    """
    Create a TRACER DirectRLEnv skeleton class after Isaac Sim runtime is active.

    This does not instantiate the env yet.
    It only verifies that TRACER can define a DirectRLEnv-compatible class.
    """

    DirectRLEnv, DirectRLEnvCfg = load_direct_env_symbols()

    class TracerDirectEnv(TracerDirectEnvLogicMixin, DirectRLEnv):
        """
        Runtime-created TRACER DirectRLEnv skeleton.

        Current M29 scope:
            - prove DirectRLEnv symbols are available after AppLauncher
            - prove TRACER can define a DirectRLEnv-compatible class
            - expose the standard DirectRLEnv method names

        Future scope:
            - M30: cfg + minimal scene
            - M31: robot articulation
            - M32: TRACER observation/action/reward wiring
        """

        cfg: DirectRLEnvCfg

        def _setup_scene(self):
            raise NotImplementedError("Scene setup will be implemented in M30/M31.")

        def _pre_physics_step(self, actions):
            raise NotImplementedError("Action preprocessing will be implemented in M30/M31.")

        def _apply_action(self):
            raise NotImplementedError("Action application will be implemented in M30/M31.")

        def _get_observations(self):
            raise NotImplementedError("TRACER observations will be implemented in M30/M31.")

        def _get_rewards(self):
            raise NotImplementedError("TRACER rewards will be implemented in M30/M31.")

        def _get_dones(self):
            raise NotImplementedError("TRACER terminations will be implemented in M30/M31.")

        def _reset_idx(self, env_ids):
            raise NotImplementedError("TRACER reset logic will be implemented in M30/M31.")

    return TracerDirectEnv, DirectRLEnv, DirectRLEnvCfg


def describe_tracer_direct_env_skeleton():
    TracerDirectEnv, DirectRLEnv, DirectRLEnvCfg = make_tracer_direct_env_class()

    required_methods = [
        "_setup_scene",
        "_pre_physics_step",
        "_apply_action",
        "_get_observations",
        "_get_rewards",
        "_get_dones",
        "_reset_idx",
    ]

    method_status = {
        name: hasattr(TracerDirectEnv, name)
        for name in required_methods
    }

    return {
        "TracerDirectEnv": TracerDirectEnv,
        "DirectRLEnv": DirectRLEnv,
        "DirectRLEnvCfg": DirectRLEnvCfg,
        "is_subclass": issubclass(TracerDirectEnv, DirectRLEnv),
        "method_status": method_status,
    }
