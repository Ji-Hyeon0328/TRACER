import torch


def load_minimal_direct_env_symbols():
    """
    Must be called after Isaac Sim runtime is launched through AppLauncher.
    """
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass

    return DirectRLEnv, DirectRLEnvCfg, InteractiveSceneCfg, SimulationCfg, configclass


def make_tracer_minimal_direct_env_class():
    """
    Build a minimal DirectRLEnv class after Isaac Sim runtime is active.

    Scope:
        - no robot yet
        - no terrain yet
        - no TRACER StepAdapter yet
        - only validates DirectRLEnv instantiate/reset/step path
    """

    (
        DirectRLEnv,
        DirectRLEnvCfg,
        InteractiveSceneCfg,
        SimulationCfg,
        configclass,
    ) = load_minimal_direct_env_symbols()

    @configclass
    class TracerMinimalDirectEnvCfg(DirectRLEnvCfg):
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
            num_envs=8,
            env_spacing=2.0,
        )

    class TracerMinimalDirectEnv(DirectRLEnv):
        cfg: TracerMinimalDirectEnvCfg

        def __init__(self, cfg: TracerMinimalDirectEnvCfg, render_mode=None, **kwargs):
            self._actions = None
            self._policy_obs = None
            super().__init__(cfg, render_mode=render_mode, **kwargs)

        def _setup_scene(self):
            # Minimal empty scene.
            # Robot and terrain will be added in M31.
            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions(global_prim_paths=[])

            try:
                import isaaclab.sim as sim_utils

                light_cfg = sim_utils.DomeLightCfg(
                    intensity=2000.0,
                    color=(0.75, 0.75, 0.75),
                )
                light_cfg.func("/World/Light", light_cfg)
            except Exception as e:
                print("[WARN] Could not create dome light:", e)

        def _pre_physics_step(self, actions):
            self._actions = actions.clone()

        def _apply_action(self):
            # No robot yet. No-op.
            pass

        def _get_observations(self):
            if self._policy_obs is None:
                self._policy_obs = torch.zeros(
                    self.num_envs,
                    self.cfg.observation_space,
                    device=self.device,
                )

            if self._actions is not None:
                # Put action into the first 12 dims just to verify step data flow.
                self._policy_obs[:, : self.cfg.action_space] = self._actions

            return {"policy": self._policy_obs}

        def _get_rewards(self):
            if self._actions is None:
                return torch.zeros(self.num_envs, device=self.device)

            return -torch.sum(self._actions * self._actions, dim=-1)

        def _get_dones(self):
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
                self._policy_obs.zero_()
            else:
                self._policy_obs[env_ids] = 0.0

    return TracerMinimalDirectEnv, TracerMinimalDirectEnvCfg
