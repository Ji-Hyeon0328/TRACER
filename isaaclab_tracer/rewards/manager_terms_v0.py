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


def tracer_motion_tracking_reward(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
) -> torch.Tensor:
    """TRACER online motion reward.

    For Isaac Lab v0, this is velocity tracking in the yaw-aligned base frame.
    This corresponds to the online form of R_motion.
    """
    asset = _robot(env, asset_cfg)
    command = _command(env, command_name)

    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(torch.square(command[:, :2] - vel_yaw[:, :2]), dim=1)

    return torch.exp(-lin_vel_error / (std**2))


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
    height_score = torch.clamp((base_z - z_fail) / max(z_good - z_fail, 1e-9), 0.0, 1.0)

    projected_gravity = _projected_gravity(asset)
    tilt_error = torch.linalg.norm(projected_gravity[:, :2], dim=1)
    orientation_score = torch.exp(-orientation_scale * tilt_error)

    return torch.clamp(0.5 * height_score + 0.5 * orientation_score, 0.0, 1.0)


def tracer_energy_reward(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    torque_scale: float = 2.0e-4,
    power_scale: float = 2.0e-4,
) -> torch.Tensor:
    """TRACER online energy reward.

    Unlike Gazebo v0, Isaac Lab can expose torque and joint velocity.
    This is still a v0 proxy, but it is closer to real energy than command-only cost.
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

    return torch.exp(-effort)


def tracer_aux_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_fail: float = 0.18,
    orientation_limit: float = 0.70,
    vertical_velocity_scale: float = 1.0,
) -> torch.Tensor:
    """Auxiliary penalty for unsafe posture and unstable vertical motion.

    This is the online counterpart of R_aux.
    It returns a non-negative penalty, not a reward.
    """
    asset = _robot(env, asset_cfg)

    base_z = asset.data.root_pos_w[:, 2]
    low_z_penalty = torch.clamp((z_fail - base_z) / max(z_fail, 1e-9), 0.0, 1.0)

    projected_gravity = _projected_gravity(asset)
    tilt_error = torch.linalg.norm(projected_gravity[:, :2], dim=1)
    orientation_penalty = torch.clamp(tilt_error / max(orientation_limit, 1e-9), 0.0, 1.0)

    vertical_vel_penalty = torch.clamp(torch.abs(asset.data.root_lin_vel_w[:, 2]) * vertical_velocity_scale, 0.0, 1.0)

    return 0.5 * low_z_penalty + 0.3 * orientation_penalty + 0.2 * vertical_vel_penalty


def tracer_slide_reward_total(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    beta_motion: float = 1.0 / 3.0,
    beta_stability: float = 1.0 / 3.0,
    beta_energy: float = 1.0 / 3.0,
    lambda_energy: float = 0.5,
    aux_scale: float = 1.5,
) -> torch.Tensor:
    """Manager-based online TRACER slide reward.

    R = (beta_m R_motion + beta_s R_stability + beta_e lambda_E R_energy) / N
        * exp(-c_aux R_aux)
    """
    r_motion = tracer_motion_tracking_reward(env, command_name=command_name, asset_cfg=asset_cfg)
    r_stability = tracer_stability_reward(env, asset_cfg=asset_cfg)
    r_energy = tracer_energy_reward(env, asset_cfg=asset_cfg)
    r_aux = tracer_aux_penalty(env, asset_cfg=asset_cfg)

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
