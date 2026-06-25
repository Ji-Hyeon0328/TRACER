from __future__ import annotations

import torch


def exp_reward(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.exp(-scale * torch.clamp(x, min=0.0))


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def motion_reward_hold(
    dx: torch.Tensor,
    dy: torch.Tensor,
    *,
    drift_scale: float = 3.0,
) -> torch.Tensor:
    """Recovery/active-hold motion reward.

    For hold/recovery primitives, less planar drift is better.
    """
    drift_xy = torch.sqrt(dx * dx + dy * dy + 1e-12)
    return clamp01(exp_reward(drift_xy, drift_scale))


def motion_reward_tracking(
    command_vx: torch.Tensor,
    observed_vx: torch.Tensor,
    dy: torch.Tensor,
    *,
    tracking_scale: float = 6.0,
    lateral_scale: float = 3.0,
) -> torch.Tensor:
    """Forward locomotion velocity tracking reward."""
    tracking_error = torch.abs(observed_vx - command_vx)
    r_track = exp_reward(tracking_error, tracking_scale)
    r_lateral = exp_reward(torch.abs(dy), lateral_scale)
    return clamp01(0.75 * r_track + 0.25 * r_lateral)


def motion_reward_backstep(
    command_vx: torch.Tensor,
    observed_vx: torch.Tensor,
    dy: torch.Tensor,
    *,
    tracking_scale: float = 8.0,
    lateral_scale: float = 4.0,
) -> torch.Tensor:
    """Backstep primitive tracking reward."""
    tracking_error = torch.abs(observed_vx - command_vx)
    r_track = exp_reward(tracking_error, tracking_scale)
    r_lateral = exp_reward(torch.abs(dy), lateral_scale)
    return clamp01(0.7 * r_track + 0.3 * r_lateral)


def stability_reward(
    min_z: torch.Tensor,
    max_abs_roll: torch.Tensor,
    max_abs_pitch: torch.Tensor,
    fall_like: torch.Tensor,
    *,
    z_fail: float = 0.18,
    z_good: float = 0.28,
    rpy_limit: float = 0.70,
) -> torch.Tensor:
    z_score = clamp01((min_z - z_fail) / max(z_good - z_fail, 1e-9))
    roll_score = clamp01(1.0 - max_abs_roll / rpy_limit)
    pitch_score = clamp01(1.0 - max_abs_pitch / rpy_limit)

    r = 0.45 * z_score + 0.25 * roll_score + 0.30 * pitch_score
    fall_scale = torch.where(fall_like > 0.5, torch.full_like(r, 0.10), torch.ones_like(r))
    return clamp01(r * fall_scale)


def energy_proxy_reward(
    command_vx: torch.Tensor,
    command_yaw: torch.Tensor,
    command_body_height: torch.Tensor,
    command_clearance: torch.Tensor,
    command_enable: torch.Tensor,
) -> torch.Tensor:
    """Gazebo-compatible energy proxy.

    Isaac Lab should later replace this with torque/joint-power energy.
    """
    effort = (
        2.0 * torch.abs(command_vx)
        + 0.5 * torch.abs(command_yaw)
        + 3.0 * torch.clamp(command_body_height - 0.305, min=0.0)
        + 4.0 * torch.clamp(command_clearance - 0.055, min=0.0)
        + 0.15 * torch.where(
            command_enable < 0.5,
            torch.ones_like(command_enable),
            torch.zeros_like(command_enable),
        )
    )
    return clamp01(exp_reward(effort, scale=2.0))


def aux_penalty(
    dx: torch.Tensor,
    dy: torch.Tensor,
    z_drop: torch.Tensor,
    min_z: torch.Tensor,
    max_abs_roll: torch.Tensor,
    max_abs_pitch: torch.Tensor,
    fall_like: torch.Tensor,
) -> torch.Tensor:
    drift_xy = torch.sqrt(dx * dx + dy * dy + 1e-12)

    fall_penalty = torch.where(fall_like > 0.5, torch.ones_like(fall_like), torch.zeros_like(fall_like))
    low_z_penalty = clamp01((0.18 - min_z) / 0.18)
    z_drop_penalty = clamp01(torch.clamp(z_drop, min=0.0) / 0.25)
    rpy_penalty = clamp01(torch.maximum(max_abs_roll, max_abs_pitch) / 0.70)
    drift_penalty = clamp01(drift_xy / 1.0)

    return (
        1.00 * fall_penalty
        + 0.60 * low_z_penalty
        + 0.40 * z_drop_penalty
        + 0.30 * rpy_penalty
        + 0.20 * drift_penalty
    )


def tracer_slide_reward(
    r_motion: torch.Tensor,
    r_stability: torch.Tensor,
    r_energy: torch.Tensor,
    r_aux: torch.Tensor,
    beta_motion: torch.Tensor,
    beta_stability: torch.Tensor,
    beta_energy: torch.Tensor,
    *,
    lambda_energy: float = 0.5,
    aux_scale: float = 1.5,
) -> torch.Tensor:
    """TRACER slide reward.

    R = (beta_m * R_motion + beta_s * R_stability + beta_e * lambda_E * R_energy) / N
        * exp(-c_aux * R_aux)
    """
    normalizer = beta_motion + beta_stability + beta_energy * lambda_energy
    normalizer = torch.clamp(normalizer, min=1e-9)

    weighted = (
        beta_motion * r_motion
        + beta_stability * r_stability
        + beta_energy * lambda_energy * r_energy
    ) / normalizer

    return weighted * torch.exp(-aux_scale * torch.clamp(r_aux, min=0.0))
