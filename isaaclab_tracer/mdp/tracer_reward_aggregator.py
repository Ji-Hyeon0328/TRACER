from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from isaaclab_tracer.mdp.tracer_rewards import (
    velocity_tracking_reward,
    yaw_tracking_reward,
    stability_reward,
    energy_penalty,
)


@dataclass
class TracerRewardOutput:
    total: torch.Tensor
    components: Dict[str, torch.Tensor]


def goal_progress_reward(prev_goal_dist: torch.Tensor, goal_dist: torch.Tensor):
    return (prev_goal_dist - goal_dist).squeeze(-1)


def success_bonus(goal_dist: torch.Tensor, tolerance: float, bonus: float):
    reached = goal_dist.squeeze(-1) < tolerance
    return reached.float() * bonus


def fall_penalty(fallen: torch.Tensor, penalty: float):
    return fallen.float() * penalty


def compute_tracer_reward_v0(
    prev_goal_dist: torch.Tensor,
    goal_dist: torch.Tensor,
    base_lin_vel_body: torch.Tensor,
    base_yaw_rate: torch.Tensor,
    tracer_cmd: torch.Tensor,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    action: torch.Tensor,
    beta: torch.Tensor,
    fallen: torch.Tensor,
    success_tolerance: float = 0.30,
):
    vx_cmd = tracer_cmd[:, 0:1]
    yaw_cmd = tracer_cmd[:, 1:2]

    v_meas = base_lin_vel_body[:, 0:1]
    yaw_rate = base_yaw_rate

    r_progress = goal_progress_reward(prev_goal_dist, goal_dist)
    r_vel = velocity_tracking_reward(v_meas, vx_cmd)
    r_yaw = yaw_tracking_reward(yaw_rate, yaw_cmd)
    r_stab = stability_reward(roll, pitch)
    r_energy = -energy_penalty(action)
    r_success = success_bonus(goal_dist, success_tolerance, 5.0)
    r_fall = fall_penalty(fallen, -5.0)

    # beta-conditioned motion/stability/energy mixture.
    beta_v = beta[:, 0]
    beta_s = beta[:, 1]
    beta_e = beta[:, 2]

    r_motion = 0.7 * r_vel + 0.3 * r_yaw
    r_beta = beta_v * r_motion + beta_s * r_stab + beta_e * r_energy

    total = (
        1.0 * r_progress
        + 0.5 * r_beta
        + r_success
        + r_fall
    )

    return TracerRewardOutput(
        total=total,
        components={
            "progress": r_progress,
            "velocity": r_vel,
            "yaw": r_yaw,
            "stability": r_stab,
            "energy": r_energy,
            "beta_mixed": r_beta,
            "success": r_success,
            "fall": r_fall,
        },
    )
