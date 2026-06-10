from __future__ import annotations

import torch


MODE_ID = {
    "nominal": 0,
    "cautious_mismatch": 1,
    "conservative_mismatch": 2,
    "conservative_posture": 3,
    "recovery": 4,
}


def build_tracer_history_feature(
    raw_vx_cmd: torch.Tensor,
    vx_cmd: torch.Tensor,
    raw_yaw_cmd: torch.Tensor,
    yaw_cmd: torch.Tensor,
    goal_distance: torch.Tensor,
    vx_tracking_error: torch.Tensor,
    yaw_tracking_error: torch.Tensor,
    base_height: torch.Tensor,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    mode_id: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            raw_vx_cmd,
            vx_cmd,
            raw_yaw_cmd,
            yaw_cmd,
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


def build_simple_context(
    height_scan: torch.Tensor,
    projected_gravity: torch.Tensor,
    base_lin_vel: torch.Tensor,
) -> torch.Tensor:
    # Minimal placeholder context.
    # Later this should become terrain/context encoder output.
    return torch.cat([height_scan, projected_gravity, base_lin_vel], dim=-1)
