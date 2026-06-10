from __future__ import annotations

import torch


def scale_joint_position_residual(
    action: torch.Tensor,
    default_joint_pos: torch.Tensor,
    residual_scale: float = 0.25,
):
    return default_joint_pos + residual_scale * torch.clamp(action, -1.0, 1.0)


def clamp_action(action: torch.Tensor, clip: float = 1.0):
    return torch.clamp(action, -clip, clip)
