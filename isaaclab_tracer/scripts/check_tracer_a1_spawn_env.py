import argparse
import os
import sys

import torch


def main():
    print("TRACER A1 spawn DirectRLEnv check")
    print("exe:", sys.executable)
    print("python:", sys.version.replace("\n", " "))
    print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""))
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))
    print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""))
    print("")

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        from isaaclab_tracer.envs.tracer_a1_spawn_env import (
            make_tracer_a1_spawn_env_class,
        )

        TracerA1SpawnEnv, TracerA1SpawnEnvCfg = make_tracer_a1_spawn_env_class()

        cfg = TracerA1SpawnEnvCfg()
        print("cfg robot prim path:", cfg.robot.prim_path)
        print("cfg num_envs:", cfg.scene.num_envs)
        print("cfg obs dim:", cfg.observation_space)
        print("cfg action dim:", cfg.action_space)
        print("")

        env = TracerA1SpawnEnv(cfg)

        print("env created")
        print("env.num_envs:", env.num_envs)
        print("env.device:", env.device)
        print("robot:", env.robot)
        print("joint names:", env.robot.data.joint_names)
        print("body names:", env.robot.data.body_names)
        print("default_joint_pos:", tuple(env.robot.data.default_joint_pos.shape))
        print("root_pos_w:", tuple(env.robot.data.root_pos_w.shape))
        print("joint_pos:", tuple(env.robot.data.joint_pos.shape))
        print("joint_vel:", tuple(env.robot.data.joint_vel.shape))
        print("")

        obs, extras = env.reset()
        print("reset obs policy:", tuple(obs["policy"].shape))
        print("reset root height mean:", env.robot.data.root_pos_w[:, 2].mean().item())
        print("reset joint pos mean:", env.robot.data.joint_pos.mean().item())
        print("")

        for i in range(10):
            actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
            out = env.step(actions)

            obs, rew, terminated, truncated, extras = out

            print("step", i)
            print("  obs policy:", tuple(obs["policy"].shape))
            print("  reward mean:", rew.mean().item())
            print("  terminated:", int(terminated.sum().item()))
            print("  truncated:", int(truncated.sum().item()))
            print("  root height mean:", env.robot.data.root_pos_w[:, 2].mean().item())
            print("  joint pos mean:", env.robot.data.joint_pos.mean().item())

        env.close()
        print("")
        print("M32 A1 spawn env passed.")

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
