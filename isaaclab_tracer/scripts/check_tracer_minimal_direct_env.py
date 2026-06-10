import argparse
import os
import sys

import torch


def main():
    print("TRACER minimal DirectRLEnv instantiate check")
    print("exe:", sys.executable)
    print("python:", sys.version.replace("\n", " "))
    print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""))
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))
    print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""))
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", ""))
    print("")

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        from isaaclab_tracer.envs.tracer_minimal_direct_env import (
            make_tracer_minimal_direct_env_class,
        )

        TracerMinimalDirectEnv, TracerMinimalDirectEnvCfg = make_tracer_minimal_direct_env_class()

        cfg = TracerMinimalDirectEnvCfg()
        print("cfg:", cfg)
        print("num_envs:", cfg.scene.num_envs)
        print("obs dim:", cfg.observation_space)
        print("action dim:", cfg.action_space)
        print("episode_length_s:", cfg.episode_length_s)
        print("")

        env = TracerMinimalDirectEnv(cfg)

        print("env created")
        print("env.num_envs:", env.num_envs)
        print("env.device:", env.device)
        print("env.max_episode_length:", env.max_episode_length)
        print("")

        obs, extras = env.reset()
        print("reset obs keys:", obs.keys())
        print("reset policy obs:", tuple(obs["policy"].shape))
        print("reset extras type:", type(extras))
        print("")

        for i in range(5):
            actions = torch.randn(env.num_envs, cfg.action_space, device=env.device) * 0.1
            out = env.step(actions)

            print("step", i)
            print("  return length:", len(out))

            obs, rew, terminated, truncated, extras = out

            print("  obs policy:", tuple(obs["policy"].shape))
            print("  reward:", tuple(rew.shape), rew.mean().item())
            print("  terminated:", tuple(terminated.shape), int(terminated.sum().item()))
            print("  truncated:", tuple(truncated.shape), int(truncated.sum().item()))
            print("  extras type:", type(extras))

        env.close()
        print("")
        print("M30 minimal DirectRLEnv instantiate passed.")

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
