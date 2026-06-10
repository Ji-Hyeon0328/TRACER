from __future__ import annotations

import torch

from isaaclab_tracer.networks.tracer_modules import TracerHighLevel


def main():
    num_envs = 4
    history_len = 40
    history_dim = 11
    context_dim = 16

    model = TracerHighLevel(
        history_dim=history_dim,
        context_dim=context_dim,
        goal_dim=2,
        latent_dim=32,
        theta_dim=6,
    )

    history = torch.randn(num_envs, history_len, history_dim)
    context = torch.randn(num_envs, context_dim)
    raw_goal_cmd = torch.randn(num_envs, 2) * 0.1

    out = model(history, context, raw_goal_cmd)

    print("rho:", out.rho.shape, out.rho.min().item(), out.rho.max().item())
    print("sigma:", out.sigma.shape, out.sigma.min().item(), out.sigma.max().item())
    print("beta:", out.beta.shape, out.beta.sum(dim=-1))
    print("theta:", out.theta.shape)
    print("vx_cmd:", out.vx_cmd.shape)
    print("yaw_cmd:", out.yaw_cmd.shape)


if __name__ == "__main__":
    main()
