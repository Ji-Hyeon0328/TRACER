from __future__ import annotations

import torch

from isaaclab_tracer.envs.tracer_task_spec import TracerTaskSpec
from isaaclab_tracer.envs.tracer_highlevel_loop import TracerHighLevelLoop


def make_fake_obs(num_envs: int, device: str, step_idx: int):
    base_xy = torch.zeros(num_envs, 2, device=device)
    base_xy[:, 0] = 0.02 * step_idx

    base_yaw = torch.zeros(num_envs, 1, device=device)

    base_lin_vel_body = torch.zeros(num_envs, 3, device=device)
    base_lin_vel_body[:, 0:1] = 0.05

    base_yaw_rate = torch.zeros(num_envs, 1, device=device)

    projected_gravity = torch.zeros(num_envs, 3, device=device)
    projected_gravity[:, 2] = -1.0

    height_scan = torch.randn(num_envs, 10, device=device) * 0.01

    base_height = torch.ones(num_envs, 1, device=device) * 0.30
    roll = torch.zeros(num_envs, 1, device=device)
    pitch = torch.zeros(num_envs, 1, device=device)

    return {
        "base_xy": base_xy,
        "base_yaw": base_yaw,
        "base_lin_vel_body": base_lin_vel_body,
        "base_yaw_rate": base_yaw_rate,
        "projected_gravity": projected_gravity,
        "height_scan": height_scan,
        "base_height": base_height,
        "roll": roll,
        "pitch": pitch,
    }


def main():
    spec = TracerTaskSpec.from_json("configs/isaac_lab/tracer_task_v0.json")

    num_envs = 8
    device = "cpu"

    loop = TracerHighLevelLoop(
        spec=spec,
        num_envs=num_envs,
        device=device,
    )

    goal_xy = torch.zeros(num_envs, 2, device=device)
    goal_xy[:, 0] = 1.0
    goal_xy[:, 1] = 0.5
    loop.set_goal_xy(goal_xy)

    for i in range(5):
        obs = make_fake_obs(num_envs, device, i)
        out = loop.step(obs)

        print("step", i)
        print("  goal_features:", tuple(out.goal_features.shape))
        print("  raw_goal_cmd:", tuple(out.raw_goal_cmd.shape))
        print("  tracer_cmd:", tuple(out.tracer_cmd.shape))
        print("  rho mean:", out.rho.mean().item())
        print("  sigma mean:", out.sigma.mean().item())
        print("  beta mean:", out.beta.mean(dim=0).detach().cpu().numpy())
        print("  theta:", tuple(out.theta.shape))
        print("  history_feature:", tuple(out.history_feature.shape))

    hist = loop.history.get()
    print("history buffer:", tuple(hist.shape))
    print("last history nonzero sum:", hist[:, -1, :].abs().sum().item())


if __name__ == "__main__":
    main()
