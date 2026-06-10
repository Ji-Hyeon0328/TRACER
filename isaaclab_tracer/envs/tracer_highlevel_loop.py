from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from isaaclab_tracer.envs.tracer_task_spec import TracerTaskSpec
from isaaclab_tracer.networks.tracer_modules import TracerHighLevel, TracerOutput
from isaaclab_tracer.utils.history_buffer import BatchedHistoryBuffer
from isaaclab_tracer.mdp.tracer_observations import (
    compute_goal_features,
    build_context_v0,
    build_history_feature_v0,
)


@dataclass
class TracerLoopOutput:
    goal_features: torch.Tensor
    raw_goal_cmd: torch.Tensor
    tracer_cmd: torch.Tensor
    rho: torch.Tensor
    sigma: torch.Tensor
    beta: torch.Tensor
    theta: torch.Tensor
    history_feature: torch.Tensor


class TracerHighLevelLoop:
    """
    Isaac-side TRACER high-level loop.

    This class mirrors the Gazebo GoalTracer loop:

        current robot observation
        -> goal tracker
        -> history-based RAM
        -> objective selector
        -> theta decoder
        -> theta-to-reference mapper
        -> vx/yaw command

    Important:
        The model predicts from the previous history buffer.
        After the finalized command is computed, the current feature is appended.
        This matches the one-step-delayed learned RAM structure used in Gazebo.
    """

    def __init__(
        self,
        spec: TracerTaskSpec,
        num_envs: int,
        device: str,
        model: Optional[TracerHighLevel] = None,
    ):
        self.spec = spec
        self.num_envs = num_envs
        self.device = device

        if model is None:
            model = TracerHighLevel(
                history_dim=spec.history_feature_dim,
                context_dim=spec.context_dim,
                goal_dim=spec.goal_dim,
                latent_dim=spec.latent_dim,
                theta_dim=spec.theta_dim,
                max_vx=spec.max_vx,
                max_yaw=spec.max_yaw,
            )

        self.model = model.to(device)

        self.history = BatchedHistoryBuffer(
            num_envs=num_envs,
            history_len=spec.history_len,
            feature_dim=spec.history_feature_dim,
            device=device,
        )

        self.goal_xy = torch.zeros(num_envs, 2, device=device)
        self.last_tracer_cmd = torch.zeros(num_envs, 2, device=device)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        self.history.reset(env_ids)

        if env_ids is None:
            self.last_tracer_cmd.zero_()
        else:
            self.last_tracer_cmd[env_ids] = 0.0

    def set_goal_xy(self, goal_xy: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            self.goal_xy[:] = goal_xy
        else:
            self.goal_xy[env_ids] = goal_xy

    def step(self, obs: Dict[str, torch.Tensor]) -> TracerLoopOutput:
        """
        Required obs keys:

            base_xy:              [B, 2]
            base_yaw:             [B, 1]
            base_lin_vel_body:    [B, 3]
            base_yaw_rate:        [B, 1]
            projected_gravity:    [B, 3]
            height_scan:          [B, K]
            base_height:          [B, 1]
            roll:                 [B, 1]
            pitch:                [B, 1]

        Optional obs keys:

            mode_id:              [B, 1]
        """

        base_xy = obs["base_xy"]
        base_yaw = obs["base_yaw"]
        base_lin_vel_body = obs["base_lin_vel_body"]
        base_yaw_rate = obs["base_yaw_rate"]
        projected_gravity = obs["projected_gravity"]
        height_scan = obs["height_scan"]
        base_height = obs["base_height"]
        roll = obs["roll"]
        pitch = obs["pitch"]

        mode_id = obs.get(
            "mode_id",
            torch.zeros(self.num_envs, 1, device=self.device),
        )

        goal_features, raw_goal_cmd = compute_goal_features(
            base_xy=base_xy,
            base_yaw=base_yaw,
            goal_xy=self.goal_xy,
            max_vx=self.spec.max_vx,
            max_yaw=self.spec.max_yaw,
        )

        context = build_context_v0(
            base_lin_vel_body=base_lin_vel_body,
            projected_gravity=projected_gravity,
            height_scan=height_scan,
            context_dim=self.spec.context_dim,
        )

        # Predict from previous history.
        out: TracerOutput = self.model(
            history=self.history.get(),
            context=context,
            raw_goal_cmd=raw_goal_cmd,
        )

        tracer_cmd = torch.cat([out.vx_cmd, out.yaw_cmd], dim=-1)

        measured_vx = base_lin_vel_body[:, 0:1]
        measured_yaw = base_yaw_rate

        vx_tracking_error = raw_goal_cmd[:, 0:1] - measured_vx
        yaw_tracking_error = raw_goal_cmd[:, 1:2] - measured_yaw

        feature = build_history_feature_v0(
            raw_goal_cmd=raw_goal_cmd,
            tracer_vx_yaw_cmd=tracer_cmd,
            goal_distance=goal_features[:, 0:1],
            vx_tracking_error=vx_tracking_error,
            yaw_tracking_error=yaw_tracking_error,
            base_height=base_height,
            roll=roll,
            pitch=pitch,
            mode_id=mode_id,
        )

        # Append finalized feature after prediction.
        self.history.append(feature)
        self.last_tracer_cmd = tracer_cmd.detach()

        return TracerLoopOutput(
            goal_features=goal_features,
            raw_goal_cmd=raw_goal_cmd,
            tracer_cmd=tracer_cmd,
            rho=out.rho,
            sigma=out.sigma,
            beta=out.beta,
            theta=out.theta,
            history_feature=feature,
        )
