from __future__ import annotations

import torch


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_goal_features(
    base_xy: torch.Tensor,
    base_yaw: torch.Tensor,
    goal_xy: torch.Tensor,
    k_v: float = 0.20,
    k_yaw: float = 0.35,
    max_vx: float = 0.12,
    max_yaw: float = 0.20,
):
    delta = goal_xy - base_xy
    dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
    desired_heading = torch.atan2(delta[:, 1:2], delta[:, 0:1])
    heading_error = wrap_to_pi(desired_heading - base_yaw)

    raw_yaw = torch.clamp(k_yaw * heading_error, -max_yaw, max_yaw)

    yaw_factor = torch.clamp(torch.cos(heading_error), min=0.0, max=1.0)
    raw_vx = torch.clamp(k_v * dist, 0.0, max_vx) * yaw_factor

    raw_goal_cmd = torch.cat([raw_vx, raw_yaw], dim=-1)
    goal_features = torch.cat([dist, heading_error], dim=-1)
    return goal_features, raw_goal_cmd


def build_context_v0(
    base_lin_vel_body: torch.Tensor,
    projected_gravity: torch.Tensor,
    height_scan: torch.Tensor,
    context_dim: int,
):
    x = torch.cat([base_lin_vel_body, projected_gravity, height_scan], dim=-1)

    if x.shape[-1] >= context_dim:
        return x[:, :context_dim]

    pad = torch.zeros(x.shape[0], context_dim - x.shape[-1], device=x.device)
    return torch.cat([x, pad], dim=-1)


def build_history_feature_v0(
    raw_goal_cmd: torch.Tensor,
    tracer_vx_yaw_cmd: torch.Tensor,
    goal_distance: torch.Tensor,
    vx_tracking_error: torch.Tensor,
    yaw_tracking_error: torch.Tensor,
    base_height: torch.Tensor,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    mode_id: torch.Tensor,
):
    raw_vx = raw_goal_cmd[:, 0:1]
    raw_yaw = raw_goal_cmd[:, 1:2]
    vx = tracer_vx_yaw_cmd[:, 0:1]
    yaw = tracer_vx_yaw_cmd[:, 1:2]

    return torch.cat(
        [
            raw_vx,
            vx,
            raw_yaw,
            yaw,
            goal_distance,
            vx_tracking_error,
            yaw_tracking_error,
            base_height,
            roll,
            pitch,
            mode_id,
        ],
        dim=-1,
    )
