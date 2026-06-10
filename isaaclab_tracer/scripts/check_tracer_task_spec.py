from __future__ import annotations

import torch

from isaaclab_tracer.envs.tracer_task_spec import TracerTaskSpec
from isaaclab_tracer.networks.tracer_modules import TracerHighLevel
from isaaclab_tracer.mdp.tracer_observations import (
    compute_goal_features,
    build_context_v0,
    build_history_feature_v0,
)
from isaaclab_tracer.mdp.tracer_terminations import goal_reached, base_fallen


def main():
    spec = TracerTaskSpec.from_json("configs/isaac_lab/tracer_task_v0.json")

    num_envs = 8
    device = "cpu"

    model = TracerHighLevel(
        history_dim=spec.history_feature_dim,
        context_dim=spec.context_dim,
        goal_dim=spec.goal_dim,
        latent_dim=spec.latent_dim,
        theta_dim=spec.theta_dim,
        max_vx=spec.max_vx,
        max_yaw=spec.max_yaw,
    ).to(device)

    base_xy = torch.zeros(num_envs, 2, device=device)
    base_yaw = torch.zeros(num_envs, 1, device=device)
    goal_xy = torch.randn(num_envs, 2, device=device)
    goal_xy[:, 0] += 1.0

    goal_features, raw_goal_cmd = compute_goal_features(
        base_xy=base_xy,
        base_yaw=base_yaw,
        goal_xy=goal_xy,
        max_vx=spec.max_vx,
        max_yaw=spec.max_yaw,
    )

    base_lin_vel_body = torch.randn(num_envs, 3, device=device) * 0.1
    projected_gravity = torch.randn(num_envs, 3, device=device)
    height_scan = torch.randn(num_envs, 10, device=device) * 0.02

    context = build_context_v0(
        base_lin_vel_body=base_lin_vel_body,
        projected_gravity=projected_gravity,
        height_scan=height_scan,
        context_dim=spec.context_dim,
    )

    history = torch.zeros(
        num_envs,
        spec.history_len,
        spec.history_feature_dim,
        device=device,
    )

    out = model(history, context, raw_goal_cmd)

    tracer_cmd = torch.cat([out.vx_cmd, out.yaw_cmd], dim=-1)

    feature = build_history_feature_v0(
        raw_goal_cmd=raw_goal_cmd,
        tracer_vx_yaw_cmd=tracer_cmd,
        goal_distance=goal_features[:, 0:1],
        vx_tracking_error=torch.zeros(num_envs, 1, device=device),
        yaw_tracking_error=torch.zeros(num_envs, 1, device=device),
        base_height=torch.ones(num_envs, 1, device=device) * 0.30,
        roll=torch.zeros(num_envs, 1, device=device),
        pitch=torch.zeros(num_envs, 1, device=device),
        mode_id=torch.zeros(num_envs, 1, device=device),
    )

    reached = goal_reached(goal_features[:, 0:1], tolerance=0.30)
    fallen = base_fallen(torch.ones(num_envs, 1, device=device) * 0.30)

    print("Task:", spec.name)
    print("goal_features:", goal_features.shape)
    print("raw_goal_cmd:", raw_goal_cmd.shape)
    print("context:", context.shape)
    print("history:", history.shape)
    print("rho:", out.rho.shape)
    print("sigma:", out.sigma.shape)
    print("beta:", out.beta.shape, out.beta.sum(dim=-1))
    print("theta:", out.theta.shape)
    print("tracer_cmd:", tracer_cmd.shape)
    print("history_feature:", feature.shape)
    print("reached:", reached.shape, reached)
    print("fallen:", fallen.shape, fallen)


if __name__ == "__main__":
    main()
