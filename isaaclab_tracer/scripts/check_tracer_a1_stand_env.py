import argparse
import os
import sys

import torch


def main():
    print("TRACER A1 stand hold single-case probe")
    print("exe:", sys.executable)
    print("python:", sys.version.replace("\n", " "))
    print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""))
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))
    print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""))
    print("")

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str, default="a1", choices=["a1", "go2"])
    parser.add_argument("--stand_stiffness", type=float, default=60.0)
    parser.add_argument("--stand_damping", type=float, default=2.0)
    parser.add_argument("--num_steps", type=int, default=120)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--disable_termination", action="store_true")
    parser.add_argument("--use_explicit_stand_pose", action="store_true")
    parser.add_argument("--stand_base_height", type=float, default=0.36)
    parser.add_argument("--stand_front_thigh", type=float, default=0.80)
    parser.add_argument("--stand_rear_thigh", type=float, default=1.00)
    parser.add_argument("--stand_calf", type=float, default=-1.50)
    parser.add_argument("--freeze_when_fallen", action="store_true")
    parser.add_argument("--freeze_height", type=float, default=0.22)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        from isaaclab_tracer.envs.tracer_a1_stand_env import (
            make_tracer_a1_stand_env_class,
        )

        EnvCls, CfgCls = make_tracer_a1_stand_env_class()

        cfg = CfgCls()
        cfg.robot_name = args_cli.robot_name
        cfg.stand_stiffness = args_cli.stand_stiffness
        cfg.stand_damping = args_cli.stand_damping
        cfg.scene.num_envs = args_cli.num_envs
        cfg.disable_termination = args_cli.disable_termination
        cfg.use_explicit_stand_pose = args_cli.use_explicit_stand_pose
        cfg.stand_base_height = args_cli.stand_base_height
        cfg.stand_front_thigh = args_cli.stand_front_thigh
        cfg.stand_rear_thigh = args_cli.stand_rear_thigh
        cfg.stand_calf = args_cli.stand_calf
        cfg.freeze_when_fallen = args_cli.freeze_when_fallen
        cfg.freeze_height = args_cli.freeze_height

        print("=" * 80)
        print("Case: robot =", cfg.robot_name, "stiffness =", cfg.stand_stiffness, "damping =", cfg.stand_damping)
        print("=" * 80)

        env = EnvCls(cfg)
        obs, extras = env.reset()

        h0 = env.robot.data.root_pos_w[:, 2].mean().item()
        print("reset root height mean:", h0)
        print("default joint pos mean:", env.robot.data.default_joint_pos.mean().item())
        print("joint pos mean:", env.robot.data.joint_pos.mean().item())
        print("joint names:", env.robot.data.joint_names)
        print("use_explicit_stand_pose:", cfg.use_explicit_stand_pose)
        print("stand_base_height:", cfg.stand_base_height)
        print("stand_front_thigh:", cfg.stand_front_thigh)
        print("stand_rear_thigh:", cfg.stand_rear_thigh)
        print("stand_calf:", cfg.stand_calf)
        print("")

        min_h = 999.0
        max_h = -999.0
        terminated_total = 0

        for i in range(args_cli.num_steps):
            actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
            obs, rew, terminated, truncated, extras = env.step(actions)

            h = env.robot.data.root_pos_w[:, 2].mean().item()
            min_h = min(min_h, h)
            max_h = max(max_h, h)
            terminated_total += int(terminated.sum().item())

            if i % 10 == 0 or i == args_cli.num_steps - 1:
                print("step", i)
                print("  height mean:", h)
                print("  reward mean:", rew.mean().item())
                print("  terminated:", int(terminated.sum().item()))
                print("  truncated:", int(truncated.sum().item()))
                print("  joint vel abs mean:", env.robot.data.joint_vel.abs().mean().item())

        final_h = env.robot.data.root_pos_w[:, 2].mean().item()

        print("")
        print("summary")
        print("  stiffness:", cfg.stand_stiffness)
        print("  damping:", cfg.stand_damping)
        print("  h0:", h0)
        print("  min_h:", min_h)
        print("  max_h:", max_h)
        print("  final_h:", final_h)
        print("  terminated_total:", terminated_total)

        env.close()
        print("")
        print("M34 A1 stand hold single-case probe completed.")

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
