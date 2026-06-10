from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from isaaclab_tracer.envs.tracer_task_spec import TracerTaskSpec
from isaaclab_tracer.envs.tracer_highlevel_loop import TracerHighLevelLoop
from isaaclab_tracer.envs.tracer_step_adapter import TracerStepAdapter
from isaaclab_tracer.mdp.tracer_policy_obs import policy_obs_dim_v0


@dataclass
class TracerMockEnvOutput:
    obs: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    extras: Dict[str, torch.Tensor]


class TracerMockEnv:
    """
    Lightweight vectorized environment skeleton for TRACER.

    This is not the final Isaac Lab environment.
    It is a minimal env-style wrapper that verifies the full TRACER loop:

        reset()
        step(action)
        -> policy_obs, reward, done, extras

    Later this logic will be moved into an Isaac Lab DirectRLEnv or
    ManagerBasedRLEnv implementation.
    """

    def __init__(
        self,
        spec: TracerTaskSpec,
        num_envs: int = 8,
        device: str = "cpu",
    ):
        self.spec = spec
        self.num_envs = num_envs
        self.device = device

        self.highlevel_loop = TracerHighLevelLoop(
            spec=spec,
            num_envs=num_envs,
            device=device,
        )

        self.default_joint_pos = torch.zeros(num_envs, 12, device=device)

        self.adapter = TracerStepAdapter(
            highlevel_loop=self.highlevel_loop,
            default_joint_pos=self.default_joint_pos,
            success_tolerance=0.30,
            residual_scale=0.25,
        )

        self.episode_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.max_episode_steps = int(spec.episode_length_sec / spec.control_dt)

        self.base_xy = torch.zeros(num_envs, 2, device=device)
        self.base_yaw = torch.zeros(num_envs, 1, device=device)
        self.previous_action = torch.zeros(num_envs, 12, device=device)

        self.goal_xy = torch.zeros(num_envs, 2, device=device)
        self.goal_xy[:, 0] = 1.0
        self.goal_xy[:, 1] = 0.5

        self.highlevel_loop.set_goal_xy(self.goal_xy)

    @property
    def observation_dim(self) -> int:
        return policy_obs_dim_v0(num_dofs=12, action_dim=12, theta_dim=self.spec.theta_dim)

    @property
    def action_dim(self) -> int:
        return 12

    def reset(self) -> torch.Tensor:
        self.episode_step.zero_()
        self.base_xy.zero_()
        self.base_yaw.zero_()
        self.previous_action.zero_()

        self.adapter.reset()
        self.highlevel_loop.set_goal_xy(self.goal_xy)

        obs_dict = self._make_obs_dict()
        zero_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        out = self.adapter.step(obs_dict, zero_action)
        return out.policy_obs.detach()

    def step(self, action: torch.Tensor) -> TracerMockEnvOutput:
        action = torch.clamp(action, -1.0, 1.0)

        # Minimal fake dynamics:
        # move base forward using previous TRACER vx command.
        # This only tests data flow. It is not physics.
        obs_dict_before = self._make_obs_dict()
        tracer_out = self.highlevel_loop.step(obs_dict_before)

        vx = tracer_out.tracer_cmd[:, 0:1].detach()
        yaw_rate = tracer_out.tracer_cmd[:, 1:2].detach()

        dt = self.spec.control_dt
        self.base_yaw = self.base_yaw + yaw_rate * dt

        dx = vx * torch.cos(self.base_yaw) * dt
        dy = vx * torch.sin(self.base_yaw) * dt
        self.base_xy = self.base_xy + torch.cat([dx, dy], dim=-1)

        self.previous_action = action.detach()
        self.episode_step += 1

        obs_dict = self._make_obs_dict()
        out = self.adapter.step(obs_dict, action)

        timeout = self.episode_step >= self.max_episode_steps
        done = torch.logical_or(out.done, timeout)

        extras = {
            "goal_distance": out.tracer.goal_features[:, 0],
            "rho": out.tracer.rho.squeeze(-1),
            "sigma": out.tracer.sigma.squeeze(-1),
            "beta_v": out.tracer.beta[:, 0],
            "beta_s": out.tracer.beta[:, 1],
            "beta_e": out.tracer.beta[:, 2],
            "vx_cmd": out.tracer.tracer_cmd[:, 0],
            "yaw_cmd": out.tracer.tracer_cmd[:, 1],
            "timeout": timeout.float(),
        }

        return TracerMockEnvOutput(
            obs=out.policy_obs.detach(),
            reward=out.reward.detach(),
            done=done.detach(),
            extras=extras,
        )

    def _make_obs_dict(self) -> Dict[str, torch.Tensor]:
        base_lin_vel_body = torch.zeros(self.num_envs, 3, device=self.device)
        base_lin_vel_body[:, 0:1] = 0.05

        base_ang_vel_body = torch.zeros(self.num_envs, 3, device=self.device)
        base_yaw_rate = torch.zeros(self.num_envs, 1, device=self.device)

        projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        projected_gravity[:, 2] = -1.0

        height_scan = torch.randn(self.num_envs, 10, device=self.device) * 0.01

        base_height = torch.ones(self.num_envs, 1, device=self.device) * 0.30
        roll = torch.zeros(self.num_envs, 1, device=self.device)
        pitch = torch.zeros(self.num_envs, 1, device=self.device)

        joint_pos = torch.zeros(self.num_envs, 12, device=self.device)
        joint_vel = torch.zeros(self.num_envs, 12, device=self.device)

        return {
            "base_xy": self.base_xy,
            "base_yaw": self.base_yaw,
            "base_lin_vel_body": base_lin_vel_body,
            "base_ang_vel_body": base_ang_vel_body,
            "base_yaw_rate": base_yaw_rate,
            "projected_gravity": projected_gravity,
            "height_scan": height_scan,
            "base_height": base_height,
            "roll": roll,
            "pitch": pitch,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "previous_action": self.previous_action,
        }
