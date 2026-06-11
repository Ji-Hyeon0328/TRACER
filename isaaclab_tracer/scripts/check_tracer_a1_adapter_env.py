import argparse
import os
import sys


def main():
    print("TRACER A1 adapter DirectRLEnv check", flush=True)
    print("exe:", sys.executable, flush=True)
    print("python:", sys.version.replace("\n", " "), flush=True)
    print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""), flush=True)
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""), flush=True)
    print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""), flush=True)
    print("", flush=True)

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--residual_scale", type=float, default=0.0)
    parser.add_argument("--num_steps", type=int, default=80)
    parser.add_argument("--hold_default_pose", action="store_true")
    parser.add_argument(
        "--action_mode",
        type=str,
        default="zero",
        choices=["zero", "constant", "random"],
    )
    parser.add_argument("--action_value", type=float, default=0.5)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    print("[M41] launching Isaac app...", flush=True)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    print("[M41] Isaac app launched.", flush=True)

    try:
        print("[M41] before import torch", flush=True)
        import torch
        print("[M41] after import torch", flush=True)

        print("[M41] before import env factory", flush=True)
        from isaaclab_tracer.envs.tracer_a1_adapter_env import (
            make_tracer_a1_adapter_env_class,
        )
        print("[M41] after import env factory", flush=True)

        print("[M41] imported env class factory.", flush=True)

        EnvCls, CfgCls = make_tracer_a1_adapter_env_class()

        cfg = CfgCls()
        cfg.residual_scale = args_cli.residual_scale
        cfg.hold_default_pose = args_cli.hold_default_pose

        print("cfg robot prim path:", cfg.robot.prim_path, flush=True)
        print("cfg num_envs:", cfg.scene.num_envs, flush=True)
        print("cfg obs dim:", cfg.observation_space, flush=True)
        print("cfg action dim:", cfg.action_space, flush=True)
        print("residual_scale:", cfg.residual_scale, flush=True)
        print("num_steps:", args_cli.num_steps, flush=True)
        print("hold_default_pose:", cfg.hold_default_pose, flush=True)
        print("action_mode:", args_cli.action_mode, flush=True)
        print("action_value:", args_cli.action_value, flush=True)
        print("", flush=True)

        env = EnvCls(cfg)
        print("env created", flush=True)
        print("env.num_envs:", env.num_envs, flush=True)
        print("env.device:", env.device, flush=True)
        print("joint names:", env.robot.data.joint_names, flush=True)
        print("body names:", env.robot.data.body_names, flush=True)
        print("", flush=True)

        obs, extras = env.reset()

        print("reset obs policy:", tuple(obs["policy"].shape), flush=True)
        print("reset root height mean:", env.robot.data.root_pos_w[:, 2].mean().item(), flush=True)
        print("reset default joint pos mean:", env.robot.data.default_joint_pos.mean().item(), flush=True)
        print("reset joint pos mean:", env.robot.data.joint_pos.mean().item(), flush=True)
        print("reset default joint pos[0]:", env.robot.data.default_joint_pos[0].detach().cpu().numpy(), flush=True)
        print("reset joint pos[0]:", env.robot.data.joint_pos[0].detach().cpu().numpy(), flush=True)
        print("", flush=True)

        min_h = 999.0
        max_h = -999.0
        terminated_total = 0
        truncated_total = 0

        for i in range(args_cli.num_steps):
            if args_cli.action_mode == "zero":
                actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
            elif args_cli.action_mode == "constant":
                actions = torch.full(
                    (env.num_envs, cfg.action_space),
                    float(args_cli.action_value),
                    device=env.device,
                )
            elif args_cli.action_mode == "random":
                actions = torch.empty(env.num_envs, cfg.action_space, device=env.device).uniform_(-1.0, 1.0)
            else:
                raise RuntimeError(f"Unknown action_mode: {args_cli.action_mode}")
            obs, rew, terminated, truncated, extras = env.step(actions)

            h = env.robot.data.root_pos_w[:, 2].mean().item()
            min_h = min(min_h, h)
            max_h = max(max_h, h)
            terminated_total += int(terminated.sum().item())
            truncated_total += int(truncated.sum().item())

            if i < 12 or i % 10 == 0 or i == args_cli.num_steps - 1:
                print("step", i, flush=True)
                print("  obs policy:", tuple(obs["policy"].shape), flush=True)
                print("  reward mean:", rew.mean().item(), flush=True)
                print("  terminated:", int(terminated.sum().item()), flush=True)
                print("  truncated:", int(truncated.sum().item()), flush=True)
                print("  root height mean:", h, flush=True)
                if hasattr(env, "_joint_pos_target") and env._joint_pos_target is not None:
                    print("  target mean:", env._joint_pos_target.mean().item(), flush=True)
                    print("  target-default abs mean:", (env._joint_pos_target - env.robot.data.default_joint_pos).abs().mean().item(), flush=True)

        final_h = env.robot.data.root_pos_w[:, 2].mean().item()

        print("", flush=True)
        print("summary", flush=True)
        print("  residual_scale:", cfg.residual_scale, flush=True)
        print("  min_h:", min_h, flush=True)
        print("  max_h:", max_h, flush=True)
        print("  final_h:", final_h, flush=True)
        print("  terminated_total:", terminated_total, flush=True)
        print("  truncated_total:", truncated_total, flush=True)
        print("", flush=True)
        print("M41 A1 adapter residual-scale check completed.", flush=True)

        env.close()

    except BaseException as e:
        print("[M41] caught exception:", repr(e), flush=True)
        raise
    finally:
        print("[M41] closing Isaac app", flush=True)
        simulation_app.close()
        print("[M41] closed Isaac app", flush=True)


if __name__ == "__main__":
    main()
