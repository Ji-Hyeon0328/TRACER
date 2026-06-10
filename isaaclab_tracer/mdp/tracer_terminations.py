from __future__ import annotations

import torch


def goal_reached(goal_distance: torch.Tensor, tolerance: float):
    return goal_distance.squeeze(-1) < tolerance


def base_fallen(base_height: torch.Tensor, min_height: float = 0.18):
    return base_height.squeeze(-1) < min_height


def attitude_failure(roll: torch.Tensor, pitch: torch.Tensor, max_angle: float = 0.8):
    return torch.logical_or(
        torch.abs(roll.squeeze(-1)) > max_angle,
        torch.abs(pitch.squeeze(-1)) > max_angle,
    )


def timeout_done(episode_step: torch.Tensor, max_episode_steps: int):
    return episode_step >= max_episode_steps
