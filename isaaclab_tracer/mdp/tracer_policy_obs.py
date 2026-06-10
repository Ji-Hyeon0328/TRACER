from __future__ import annotations

import torch


def build_policy_obs_v0(
    base_lin_vel_body: torch.Tensor,
    base_ang_vel_body: torch.Tensor,
    projected_gravity: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    previous_action: torch.Tensor,
    tracer_cmd: torch.Tensor,
    beta: torch.Tensor,
    rho: torch.Tensor,
    sigma: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    Build low-level policy observation.

    This observation intentionally includes TRACER high-level outputs:

        tracer_cmd = [vx_cmd, yaw_cmd]
        beta       = [beta_v, beta_s, beta_e]
        rho/sigma  = adaptation estimates
        theta      = meta reference parameters

    The low-level policy can learn to condition locomotion on TRACER outputs.
    """

    return torch.cat(
        [
            base_lin_vel_body,
            base_ang_vel_body,
            projected_gravity,
            joint_pos,
            joint_vel,
            previous_action,
            tracer_cmd,
            beta,
            rho,
            sigma,
            theta,
        ],
        dim=-1,
    )


def policy_obs_dim_v0(num_dofs: int = 12, action_dim: int = 12, theta_dim: int = 6) -> int:
    return (
        3  # base_lin_vel_body
        + 3  # base_ang_vel_body
        + 3  # projected_gravity
        + num_dofs  # joint_pos
        + num_dofs  # joint_vel
        + action_dim  # previous_action
        + 2  # tracer_cmd
        + 3  # beta
        + 1  # rho
        + 1  # sigma
        + theta_dim
    )
