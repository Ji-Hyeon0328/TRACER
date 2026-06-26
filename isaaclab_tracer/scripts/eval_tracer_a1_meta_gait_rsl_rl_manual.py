#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from isaaclab.app import AppLauncher


def get_policy_obs(obs):
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        if "actor" in obs:
            return obs["actor"]
        raise KeyError(f"obs dict keys={list(obs.keys())}")
    return obs


def fmt_vec(x):
    vals = x.detach().flatten().cpu().tolist()
    return "[" + ", ".join(f"{float(v):+.5f}" for v in vals) + "]"


def effective_theta_for_print(action, theta_dim=6):
    import os

    active_raw = os.environ.get("TRACER_META_ACTIVE_THETA", "")
    scale_raw = os.environ.get("TRACER_META_ACTIVE_THETA_SCALE", "")

    if active_raw.strip():
        active_indices = [int(x.strip()) for x in active_raw.split(",") if x.strip() != ""]
        if scale_raw.strip():
            scales = [float(x.strip()) for x in scale_raw.split(",") if x.strip() != ""]
        else:
            scales = [1.0 for _ in active_indices]

        full = action.new_zeros((action.shape[0], theta_dim))
        for src_i, theta_i in enumerate(active_indices):
            full[:, theta_i] = action[:, src_i] * scales[src_i]
        return full

    if action.shape[-1] < theta_dim:
        full = action.new_zeros((action.shape[0], theta_dim))
        full[:, : action.shape[-1]] = action
        return full

    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Isaac-TRACER-A1-MetaGait-v0")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=320)
    parser.add_argument("--print_every", type=int, default=80)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import torch
        import torch.nn as nn

        from isaaclab_tasks.utils import parse_env_cfg

        import isaaclab_tracer.tasks.tracer_a1_meta_gait  # noqa: F401

        ckpt_path = Path(args.checkpoint).expanduser().resolve()
        print("[manual-eval] task:", args.task, flush=True)
        print("[manual-eval] checkpoint:", ckpt_path, flush=True)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print("[manual-eval] checkpoint keys:", list(ckpt.keys()), flush=True)

        if "actor_state_dict" not in ckpt:
            raise KeyError(f"actor_state_dict not found. keys={list(ckpt.keys())}")

        actor_sd = ckpt["actor_state_dict"]
        print("[manual-eval] actor keys:", list(actor_sd.keys()), flush=True)

        # rsl_rl actor from training log:
        # MLP: 58 -> 128 -> 128 -> 6 with ELU
        w0 = actor_sd["mlp.0.weight"]
        b0 = actor_sd["mlp.0.bias"]
        w1 = actor_sd["mlp.2.weight"]
        b1 = actor_sd["mlp.2.bias"]
        w2 = actor_sd["mlp.4.weight"]
        b2 = actor_sd["mlp.4.bias"]

        actor = nn.Sequential(
            nn.Linear(w0.shape[1], w0.shape[0]),
            nn.ELU(),
            nn.Linear(w1.shape[1], w1.shape[0]),
            nn.ELU(),
            nn.Linear(w2.shape[1], w2.shape[0]),
        ).to(args.device)

        with torch.no_grad():
            actor[0].weight.copy_(w0.to(args.device))
            actor[0].bias.copy_(b0.to(args.device))
            actor[2].weight.copy_(w1.to(args.device))
            actor[2].bias.copy_(b1.to(args.device))
            actor[4].weight.copy_(w2.to(args.device))
            actor[4].bias.copy_(b2.to(args.device))

        actor.eval()
        print("[manual-eval] actor loaded", flush=True)

        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=True,
        )

        print("[manual-eval] before gym.make", flush=True)
        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        print("[manual-eval] after gym.make", flush=True)

        env_unwrapped = env.unwrapped

        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        obs = get_policy_obs(obs)
        print("[manual-eval] after reset obs shape:", tuple(obs.shape), flush=True)
        cfg_obj = getattr(getattr(env, "unwrapped", env), "cfg", None)
        progress_sign = float(getattr(cfg_obj, "meta_progress_sign", 1.0))

        x0 = env_unwrapped.robot.data.root_pos_w[:, 0].detach().clone()
        y0 = env_unwrapped.robot.data.root_pos_w[:, 1].detach().clone()
        min_h = env_unwrapped.robot.data.root_pos_w[:, 2].detach().clone()

        rewards = []
        theta0_values = []
        action_values = []
        effective_theta_values = []
        done_count = 0

        for i in range(int(args.num_steps)):
            with torch.no_grad():
                action = actor(obs)
                action = torch.clamp(action, -1.0, 1.0)

            theta0_values.append(action[:, 0].detach().clone())
            action_values.append(action.detach().clone())
            effective_theta = effective_theta_for_print(action)
            effective_theta_values.append(effective_theta.detach().clone())

            step_out = env.step(action)
            if len(step_out) == 5:
                obs_next, rew, terminated, truncated, info = step_out
                dones = torch.logical_or(terminated, truncated)
            elif len(step_out) == 4:
                obs_next, rew, dones, info = step_out
            else:
                raise RuntimeError(f"Unexpected env.step output length: {len(step_out)}")

            obs = get_policy_obs(obs_next)

            rewards.append(rew.detach().clone())
            done_count += int(dones.sum().item())
            min_h = torch.minimum(min_h, env_unwrapped.robot.data.root_pos_w[:, 2].detach())

            if i == 0 or (i + 1) % int(args.print_every) == 0 or i == int(args.num_steps) - 1:
                dx_now = env_unwrapped.robot.data.root_pos_w[:, 0] - x0
                dy_now = env_unwrapped.robot.data.root_pos_w[:, 1] - y0
                print(
                    f"[manual-eval] step {i + 1}/{int(args.num_steps)} "
                    f"rew={rew.mean().item():+.5f} "
                    f"dx={dx_now.mean().item():+.5f} "
                    f"dir_dx={progress_sign * dx_now.mean().item():+.5f} "
                    f"dy={dy_now.mean().item():+.5f} "
                    f"h={env_unwrapped.robot.data.root_pos_w[:, 2].mean().item():.5f} "
                    f"done_sum={int(dones.sum().item())} "
                    f"theta0={action[:, 0].mean().item():+.5f} "
                    f"action_mean={fmt_vec(action.mean(dim=0))} "
                    f"effective_theta_mean={fmt_vec(effective_theta.mean(dim=0))}",
                    flush=True,
                )

        dx = env_unwrapped.robot.data.root_pos_w[:, 0] - x0
        dy = env_unwrapped.robot.data.root_pos_w[:, 1] - y0

        rew_all = torch.stack(rewards, dim=0)
        theta0_all = torch.stack(theta0_values, dim=0)
        action_all = torch.stack(action_values, dim=0)
        action_mean = action_all.mean(dim=(0, 1))
        action_min = action_all.amin(dim=(0, 1))
        action_max = action_all.amax(dim=(0, 1))

        effective_theta_all = torch.stack(effective_theta_values, dim=0)
        effective_theta_mean = effective_theta_all.mean(dim=(0, 1))
        effective_theta_min = effective_theta_all.amin(dim=(0, 1))
        effective_theta_max = effective_theta_all.amax(dim=(0, 1))

        print("summary", flush=True)
        print(f"  reward_mean: {rew_all.mean().item():+.6f}", flush=True)
        print(f"  reward_min: {rew_all.min().item():+.6f}", flush=True)
        print(f"  reward_max: {rew_all.max().item():+.6f}", flush=True)
        print(f"  final_dx: {dx.mean().item():+.6f}", flush=True)
        print(f"  directional_dx: {progress_sign * dx.mean().item():+.6f}", flush=True)
        print(f"  final_dy: {dy.mean().item():+.6f}", flush=True)
        print(f"  abs_final_dy: {dy.abs().mean().item():+.6f}", flush=True)
        print(f"  min_h: {min_h.min().item():.6f}", flush=True)
        print(f"  final_h: {env_unwrapped.robot.data.root_pos_w[:, 2].mean().item():.6f}", flush=True)
        print(f"  done_count: {done_count}", flush=True)
        print(f"  mean_theta0: {theta0_all.mean().item():+.6f}", flush=True)
        print(f"  min_theta0: {theta0_all.min().item():+.6f}", flush=True)
        print(f"  max_theta0: {theta0_all.max().item():+.6f}", flush=True)
        print(f"  action_dim: {action_all.shape[-1]}", flush=True)
        print(f"  action_mean: {fmt_vec(action_mean)}", flush=True)
        print(f"  action_min: {fmt_vec(action_min)}", flush=True)
        print(f"  action_max: {fmt_vec(action_max)}", flush=True)
        print(f"  effective_theta_mean: {fmt_vec(effective_theta_mean)}", flush=True)
        print(f"  effective_theta_min: {fmt_vec(effective_theta_min)}", flush=True)
        print(f"  effective_theta_max: {fmt_vec(effective_theta_max)}", flush=True)

        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
