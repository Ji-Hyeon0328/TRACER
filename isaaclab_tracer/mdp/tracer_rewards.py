from __future__ import annotations

import torch


def velocity_tracking_reward(v_meas: torch.Tensor, v_cmd: torch.Tensor, sigma: float = 0.25):
    err = torch.sum((v_meas - v_cmd) ** 2, dim=-1)
    return torch.exp(-err / sigma)


def yaw_tracking_reward(yaw_rate: torch.Tensor, yaw_cmd: torch.Tensor, sigma: float = 0.25):
    err = (yaw_rate - yaw_cmd).pow(2).squeeze(-1)
    return torch.exp(-err / sigma)


def stability_reward(roll: torch.Tensor, pitch: torch.Tensor, sigma: float = 0.25):
    err = roll.squeeze(-1).pow(2) + pitch.squeeze(-1).pow(2)
    return torch.exp(-err / sigma)


def energy_penalty(action: torch.Tensor):
    return torch.sum(action.pow(2), dim=-1)


def beta_weighted_reward(
    beta: torch.Tensor,
    r_motion: torch.Tensor,
    r_stability: torch.Tensor,
    r_energy: torch.Tensor,
):
    beta_v = beta[:, 0]
    beta_s = beta[:, 1]
    beta_e = beta[:, 2]
    return beta_v * r_motion + beta_s * r_stability + beta_e * r_energy
