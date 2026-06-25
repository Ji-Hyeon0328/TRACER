#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=320)
    parser.add_argument("--action_mode", type=str, default="zero", choices=["zero", "small_random", "vx_pos"])
    parser.add_argument("--action_scale", type=float, default=0.15)

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    print("[reward-smoke] launching Isaac app", flush=True)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    print("[reward-smoke] Isaac app launched", flush=True)

    try:
        from isaaclab_tracer.envs.tracer_a1_adapter_env import make_tracer_a1_adapter_env_class

        print("[reward-smoke] making env class", flush=True)
        TracerA1AdapterEnv, TracerA1AdapterEnvCfg = make_tracer_a1_adapter_env_class()

        cfg = TracerA1AdapterEnvCfg()
        cfg.scene.num_envs = int(args_cli.num_envs)
        cfg.sim.device = args_cli.device

        cfg.action_type = "meta_gait_theta"
        cfg.action_space = int(getattr(cfg, "meta_theta_dim", 6))
        cfg.ignore_adapter_done = True
        cfg.meta_debug = False

        # Match the stable internal meta-gait baseline branch.
        cfg.use_external_lowlevel = False
        cfg.external_lowlevel_timeout_s = 0.20
        cfg.external_lowlevel_env_index = 0
        cfg.use_grf_torque = False
        cfg.use_nominal_gait = True
        cfg.hold_default_pose = False
        cfg.residual_scale = 0.0

        print("[reward-smoke] cfg summary", flush=True)
        print("  num_envs:", cfg.scene.num_envs, flush=True)
        print("  device:", cfg.sim.device, flush=True)
        print("  action_type:", cfg.action_type, flush=True)
        print("  action_space:", cfg.action_space, flush=True)
        print("  gait_clearance2:", getattr(cfg, "gait_clearance2", None), flush=True)
        print("  gait_warmup_steps:", getattr(cfg, "gait_warmup_steps", None), flush=True)
        print("  gait_counter_speed:", getattr(cfg, "gait_counter_speed", None), flush=True)
        print("  gait_pattern:", getattr(cfg, "gait_pattern", None), flush=True)
        print("  gait_foot_delta_x_limit:", getattr(cfg, "gait_foot_delta_x_limit", None), flush=True)
        print("  gait_foot_delta_y_limit:", getattr(cfg, "gait_foot_delta_y_limit", None), flush=True)
        print("  meta_base_vx:", getattr(cfg, "meta_base_vx", None), flush=True)
        print("  meta_gait_x_sign:", getattr(cfg, "meta_gait_x_sign", None), flush=True)
        print("  meta_stance_push_gain:", getattr(cfg, "meta_stance_push_gain", None), flush=True)
        print("  meta_stance_ik_blend:", getattr(cfg, "meta_stance_ik_blend", None), flush=True)

        print("[reward-smoke] creating env", flush=True)
        env = TracerA1AdapterEnv(cfg)
        print("[reward-smoke] env created", flush=True)

        obs, _ = env.reset()
        print("[reward-smoke] env reset done", flush=True)

        rew_hist = []
        h_hist = []
        dx0 = env.robot.data.root_pos_w[:, 0].detach().clone()
        dy0 = env.robot.data.root_pos_w[:, 1].detach().clone()

        print("[reward-smoke] entering loop", flush=True)

        for i in range(int(args_cli.num_steps)):
            if args_cli.action_mode == "zero":
                actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
            elif args_cli.action_mode == "vx_pos":
                actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
                actions[:, 0] = float(args_cli.action_scale)
            else:
                actions = float(args_cli.action_scale) * torch.randn(env.num_envs, cfg.action_space, device=env.device)

            obs, rew, terminated, truncated, info = env.step(actions)

            rew_hist.append(rew.detach().clone())
            h_hist.append(env.robot.data.root_pos_w[:, 2].detach().clone())

            if i in [0, 40, 80, 120, 160, 240, int(args_cli.num_steps) - 1]:
                dx = env.robot.data.root_pos_w[:, 0] - dx0
                dy = env.robot.data.root_pos_w[:, 1] - dy0
                print(f"step {i}", flush=True)
                print("  reward mean:", float(rew.mean().item()), flush=True)
                print("  reward min/max:", float(rew.min().item()), float(rew.max().item()), flush=True)
                print("  root height mean:", float(env.robot.data.root_pos_w[:, 2].mean().item()), flush=True)
                print("  dx mean:", float(dx.mean().item()), flush=True)
                print("  dy mean:", float(dy.mean().item()), flush=True)
                print("  terminated count:", int(terminated.sum().item()), flush=True)
                print("  truncated count:", int(truncated.sum().item()), flush=True)

            if not torch.isfinite(rew).all():
                raise RuntimeError(f"Non-finite reward at step {i}: {rew}")

        rew_all = torch.stack(rew_hist, dim=0)
        h_all = torch.stack(h_hist, dim=0)
        dx = env.robot.data.root_pos_w[:, 0] - dx0
        dy = env.robot.data.root_pos_w[:, 1] - dy0

        print("summary", flush=True)
        print("  reward_mean:", float(rew_all.mean().item()), flush=True)
        print("  reward_min:", float(rew_all.min().item()), flush=True)
        print("  reward_max:", float(rew_all.max().item()), flush=True)
        print("  min_h:", float(h_all.min().item()), flush=True)
        print("  final_h:", float(env.robot.data.root_pos_w[:, 2].mean().item()), flush=True)
        print("  final_dx:", float(dx.mean().item()), flush=True)
        print("  final_dy:", float(dy.mean().item()), flush=True)

        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
