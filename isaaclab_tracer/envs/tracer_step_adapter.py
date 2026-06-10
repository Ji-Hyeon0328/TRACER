from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from isaaclab_tracer.envs.tracer_highlevel_loop import TracerHighLevelLoop, TracerLoopOutput
from isaaclab_tracer.mdp.tracer_policy_obs import build_policy_obs_v0
from isaaclab_tracer.mdp.tracer_actions import scale_joint_position_residual
from isaaclab_tracer.mdp.tracer_reward_aggregator import (
    compute_tracer_reward_v0,
    TracerRewardOutput,
)
from isaaclab_tracer.mdp.tracer_terminations import (
    goal_reached,
    base_fallen,
    attitude_failure,
)


@dataclass
class TracerStepOutput:
    policy_obs: torch.Tensor
    joint_pos_target: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    tracer: TracerLoopOutput
    reward_info: Dict[str, torch.Tensor]


class TracerStepAdapter:
    """
    Adapter that combines:

        Isaac observation dict
        + TRACER high-level loop
        + low-level policy observation construction
        + action scaling
        + reward/termination computation

    This is not a full Isaac Lab environment yet.
    It is the reusable logic that will be inserted into the real environment.
    """

    def __init__(
        self,
        highlevel_loop: TracerHighLevelLoop,
        default_joint_pos: torch.Tensor,
        success_tolerance: float = 0.30,
        residual_scale: float = 0.25,
    ):
        self.highlevel_loop = highlevel_loop
        self.default_joint_pos = default_joint_pos
        self.success_tolerance = success_tolerance
        self.residual_scale = residual_scale

        self.prev_goal_dist = None

    def reset(self, env_ids=None):
        self.highlevel_loop.reset(env_ids)
        self.prev_goal_dist = None

    def step(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> TracerStepOutput:
        tracer_out = self.highlevel_loop.step(obs)

        goal_dist = tracer_out.goal_features[:, 0:1]

        if self.prev_goal_dist is None:
            prev_goal_dist = goal_dist.detach()
        else:
            prev_goal_dist = self.prev_goal_dist

        policy_obs = build_policy_obs_v0(
            base_lin_vel_body=obs["base_lin_vel_body"],
            base_ang_vel_body=obs["base_ang_vel_body"],
            projected_gravity=obs["projected_gravity"],
            joint_pos=obs["joint_pos"],
            joint_vel=obs["joint_vel"],
            previous_action=obs["previous_action"],
            tracer_cmd=tracer_out.tracer_cmd,
            beta=tracer_out.beta,
            rho=tracer_out.rho,
            sigma=tracer_out.sigma,
            theta=tracer_out.theta,
        )

        joint_pos_target = scale_joint_position_residual(
            action=action,
            default_joint_pos=self.default_joint_pos,
            residual_scale=self.residual_scale,
        )

        fallen = base_fallen(obs["base_height"])
        attitude_fail = attitude_failure(obs["roll"], obs["pitch"])
        reached = goal_reached(goal_dist, self.success_tolerance)
        done = torch.logical_or(torch.logical_or(fallen, attitude_fail), reached)

        reward_out: TracerRewardOutput = compute_tracer_reward_v0(
            prev_goal_dist=prev_goal_dist,
            goal_dist=goal_dist,
            base_lin_vel_body=obs["base_lin_vel_body"],
            base_yaw_rate=obs["base_yaw_rate"],
            tracer_cmd=tracer_out.tracer_cmd,
            roll=obs["roll"],
            pitch=obs["pitch"],
            action=action,
            beta=tracer_out.beta,
            fallen=torch.logical_or(fallen, attitude_fail),
            success_tolerance=self.success_tolerance,
        )

        self.prev_goal_dist = goal_dist.detach()

        return TracerStepOutput(
            policy_obs=policy_obs,
            joint_pos_target=joint_pos_target,
            reward=reward_out.total,
            done=done,
            tracer=tracer_out,
            reward_info=reward_out.components,
        )
