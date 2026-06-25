from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from isaaclab_tracer.rewards.slide_reward_v0 import tracer_slide_reward

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _robot(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name]


def _command(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def _projected_gravity(asset) -> torch.Tensor:
    gravity_w = torch.zeros_like(asset.data.root_lin_vel_w[:, :3])
    gravity_w[:, 2] = -1.0
    return quat_apply_inverse(asset.data.root_quat_w, gravity_w)


def _base_velocity_yaw(asset) -> torch.Tensor:
    """Root velocity expressed in the yaw-aligned base frame."""
    return quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])


def _command_activity_progress(
    env: "ManagerBasedRLEnv",
    asset,
    command_name: str,
    *,
    active_min_speed: float = 0.10,
    active_full_speed: float = 0.50,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return command, yaw-frame velocity, command activity, progress, command speed, actual speed.

    progress is directional:
      1.0 means actual horizontal velocity matches the command direction and magnitude.
      0.0 means no progress or movement opposite to the command.
    """
    command = _command(env, command_name)
    vel_yaw = _base_velocity_yaw(asset)

    cmd_xy = command[:, :2]
    vel_xy = vel_yaw[:, :2]

    cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
    actual_speed = torch.linalg.norm(vel_xy, dim=1)

    progress = torch.sum(vel_xy * cmd_xy, dim=1) / (cmd_speed * cmd_speed + eps)
    progress = torch.clamp(progress, 0.0, 1.0)

    denom = max(active_full_speed - active_min_speed, eps)
    cmd_active = torch.clamp((cmd_speed - active_min_speed) / denom, 0.0, 1.0)

    return command, vel_yaw, cmd_active, progress, cmd_speed, actual_speed


def tracer_motion_tracking_reward(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.35,
    active_min_speed: float = 0.10,
    active_full_speed: float = 0.50,
    progress_floor: float = 0.05,
) -> torch.Tensor:
    """TRACER online motion reward with command-active anti-abandonment gating.

    V0 gave non-trivial reward to standing under forward commands.
    V1 keeps the original tracking form, but gates active-command motion by
    directional progress so that pose-hold is not a good solution when a
    non-zero velocity command is given.
    """
    asset = _robot(env, asset_cfg)
    command, vel_yaw, cmd_active, progress, _, _ = _command_activity_progress(
        env,
        asset,
        command_name,
        active_min_speed=active_min_speed,
        active_full_speed=active_full_speed,
    )

    lin_vel_error = torch.sum(torch.square(command[:, :2] - vel_yaw[:, :2]), dim=1)
    r_track = torch.exp(-lin_vel_error / max(std**2, 1.0e-9))

    progress_gate = progress_floor + (1.0 - progress_floor) * progress
    r_active = r_track * progress_gate

    return torch.clamp((1.0 - cmd_active) * r_track + cmd_active * r_active, 0.0, 1.0)


def tracer_stability_reward(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_fail: float = 0.18,
    z_good: float = 0.28,
    orientation_scale: float = 4.0,
) -> torch.Tensor:
    """TRACER online stability reward.

    Uses base height and projected-gravity tilt proxy.
    Later this can be extended with contact state and terrain-aware support margin.
    """
    asset = _robot(env, asset_cfg)

    base_z = asset.data.root_pos_w[:, 2]
    height_score = torch.clamp((base_z - z_fail) / max(z_good - z_fail, 1.0e-9), 0.0, 1.0)

    projected_gravity = _projected_gravity(asset)
    tilt_error = torch.linalg.norm(projected_gravity[:, :2], dim=1)
    orientation_score = torch.exp(-orientation_scale * tilt_error)

    return torch.clamp(0.5 * height_score + 0.5 * orientation_score, 0.0, 1.0)


def tracer_energy_reward(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    torque_scale: float = 2.0e-4,
    power_scale: float = 2.0e-4,
    active_min_speed: float = 0.10,
    active_full_speed: float = 0.50,
    progress_floor: float = 0.05,
) -> torch.Tensor:
    """TRACER online energy reward with progress gating.

    Energy efficiency should mean low energy while accomplishing the commanded
    motion, not zero energy by refusing to move.
    """
    asset = _robot(env, asset_cfg)

    joint_vel = asset.data.joint_vel
    torque = getattr(asset.data, "applied_torque", None)

    if torque is None:
        effort = torch.zeros(joint_vel.shape[0], device=joint_vel.device)
    else:
        torque_l2 = torch.mean(torch.square(torque), dim=1)
        joint_power = torch.mean(torch.abs(torque * joint_vel), dim=1)
        effort = torque_scale * torque_l2 + power_scale * joint_power

    r_energy_raw = torch.exp(-effort)

    _, _, cmd_active, progress, _, _ = _command_activity_progress(
        env,
        asset,
        command_name,
        active_min_speed=active_min_speed,
        active_full_speed=active_full_speed,
    )
    progress_gate = progress_floor + (1.0 - progress_floor) * progress

    return torch.clamp((1.0 - cmd_active) * r_energy_raw + cmd_active * r_energy_raw * progress_gate, 0.0, 1.0)


def tracer_aux_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_fail: float = 0.18,
    orientation_limit: float = 0.70,
    vertical_velocity_scale: float = 1.0,
    active_min_speed: float = 0.10,
    active_full_speed: float = 0.50,
    hold_speed_threshold: float = 0.15,
    active_hold_penalty_weight: float = 0.80,
    no_progress_penalty_weight: float = 0.40,
) -> torch.Tensor:
    """Auxiliary penalty for unsafe posture, unstable vertical motion, and active-command hold.

    This is the online counterpart of R_aux.
    It returns a non-negative penalty, not a reward.
    """
    asset = _robot(env, asset_cfg)

    base_z = asset.data.root_pos_w[:, 2]
    low_z_penalty = torch.clamp((z_fail - base_z) / max(z_fail, 1.0e-9), 0.0, 1.0)

    projected_gravity = _projected_gravity(asset)
    tilt_error = torch.linalg.norm(projected_gravity[:, :2], dim=1)
    orientation_penalty = torch.clamp(tilt_error / max(orientation_limit, 1.0e-9), 0.0, 1.0)

    vertical_vel_penalty = torch.clamp(torch.abs(asset.data.root_lin_vel_w[:, 2]) * vertical_velocity_scale, 0.0, 1.0)

    _, _, cmd_active, progress, _, actual_speed = _command_activity_progress(
        env,
        asset,
        command_name,
        active_min_speed=active_min_speed,
        active_full_speed=active_full_speed,
    )

    hold_penalty = torch.clamp((hold_speed_threshold - actual_speed) / max(hold_speed_threshold, 1.0e-9), 0.0, 1.0)
    active_hold_penalty = active_hold_penalty_weight * cmd_active * hold_penalty
    no_progress_penalty = no_progress_penalty_weight * cmd_active * (1.0 - progress)

    return (
        0.5 * low_z_penalty
        + 0.3 * orientation_penalty
        + 0.2 * vertical_vel_penalty
        + active_hold_penalty
        + no_progress_penalty
    )


def tracer_slide_reward_total(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    beta_motion: float = 1.0 / 3.0,
    beta_stability: float = 1.0 / 3.0,
    beta_energy: float = 1.0 / 3.0,
    lambda_energy: float = 0.5,
    aux_scale: float = 1.5,
    motion_std: float = 0.35,
    active_min_speed: float = 0.10,
    active_full_speed: float = 0.50,
    progress_floor: float = 0.05,
    active_hold_penalty_weight: float = 0.80,
    no_progress_penalty_weight: float = 0.40,
) -> torch.Tensor:
    """Manager-based online TRACER slide reward.

    Same high-level reward form as V0:

        R = (beta_m R_motion + beta_s R_stability + beta_e lambda_E R_energy) / N
            * exp(-c_aux R_aux)

    V1 refinement:
      - R_motion is command-active and progress-gated.
      - R_energy is progress-gated under active commands.
      - R_aux includes active-command hold/no-progress penalties.
    """
    r_motion = tracer_motion_tracking_reward(
        env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        std=motion_std,
        active_min_speed=active_min_speed,
        active_full_speed=active_full_speed,
        progress_floor=progress_floor,
    )
    r_stability = tracer_stability_reward(env, asset_cfg=asset_cfg)
    r_energy = tracer_energy_reward(
        env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        active_min_speed=active_min_speed,
        active_full_speed=active_full_speed,
        progress_floor=progress_floor,
    )
    r_aux = tracer_aux_penalty(
        env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        active_min_speed=active_min_speed,
        active_full_speed=active_full_speed,
        active_hold_penalty_weight=active_hold_penalty_weight,
        no_progress_penalty_weight=no_progress_penalty_weight,
    )

    beta_m = torch.full_like(r_motion, float(beta_motion))
    beta_s = torch.full_like(r_motion, float(beta_stability))
    beta_e = torch.full_like(r_motion, float(beta_energy))

    return tracer_slide_reward(
        r_motion,
        r_stability,
        r_energy,
        r_aux,
        beta_m,
        beta_s,
        beta_e,
        lambda_energy=lambda_energy,
        aux_scale=aux_scale,
    )


def tracer_command_progress_reward(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_min_speed: float = 0.10,
    active_full_speed: float = 0.50,
) -> torch.Tensor:
    """Explicit command-progress reward.

    This term is intentionally separate from tracer_slide_reward so that
    PPO cannot obtain high return by simply holding a stable pose under
    non-zero velocity commands.
    """
    asset = _robot(env, asset_cfg)
    _, _, cmd_active, progress, _, _ = _command_activity_progress(
        env,
        asset,
        command_name,
        active_min_speed=active_min_speed,
        active_full_speed=active_full_speed,
    )
    return torch.clamp(cmd_active * progress, 0.0, 1.0)


def tracer_active_hold_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_min_speed: float = 0.10,
    active_full_speed: float = 0.50,
    hold_speed_threshold: float = 0.20,
) -> torch.Tensor:
    """Explicit active-command hold penalty.

    Returns a positive penalty. Use a negative RewardTerm weight.
    """
    asset = _robot(env, asset_cfg)
    _, _, cmd_active, _, _, actual_speed = _command_activity_progress(
        env,
        asset,
        command_name,
        active_min_speed=active_min_speed,
        active_full_speed=active_full_speed,
    )

    hold = torch.clamp(
        (hold_speed_threshold - actual_speed) / max(hold_speed_threshold, 1.0e-9),
        0.0,
        1.0,
    )
    return torch.clamp(cmd_active * hold, 0.0, 1.0)

