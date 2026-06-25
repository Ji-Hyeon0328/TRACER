from __future__ import annotations

import argparse
import traceback

import torch

from isaaclab.app import AppLauncher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="Isaac-TRACER-Velocity-Flat-Unitree-Go1-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    env = None
    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import isaaclab_tracer.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_tracer.rewards.manager_terms_v0 import tracer_slide_reward_total

        print("==========", args.env_id, "==========")

        env_cfg = parse_env_cfg(
            args.env_id,
            device=args.device,
            num_envs=args.num_envs,
        )

        print("[TRACER] cfg parsed")
        print("[TRACER] cfg class:", type(env_cfg))
        print("[TRACER] num_envs:", env_cfg.scene.num_envs)
        print("[TRACER] tracer_slide_reward:", getattr(env_cfg.rewards, "tracer_slide_reward", None))

        env = gym.make(args.env_id, cfg=env_cfg)
        unwrapped = env.unwrapped

        print("[TRACER] env made:", args.env_id)
        print("[TRACER] device:", unwrapped.device)
        print("[TRACER] num_envs:", unwrapped.num_envs)
        print("[TRACER] action_dim:", unwrapped.action_manager.total_action_dim)

        obs, info = env.reset()
        print("[TRACER] reset ok")
        if isinstance(obs, dict):
            print("[TRACER] obs keys:", list(obs.keys()))

        action_dim = unwrapped.action_manager.total_action_dim
        action = torch.zeros((unwrapped.num_envs, action_dim), device=unwrapped.device)

        for i in range(args.steps):
            obs, rew, terminated, truncated, info = env.step(action)

            tracer_r = tracer_slide_reward_total(unwrapped)
            if not torch.isfinite(tracer_r).all():
                raise RuntimeError("TRACER slide reward produced NaN or Inf")

            print(
                f"[TRACER] step {i} ok",
                "total_rew_mean=", float(rew.mean().item()),
                "tracer_rew_mean=", float(tracer_r.mean().item()),
                "tracer_rew_min=", float(tracer_r.min().item()),
                "tracer_rew_max=", float(tracer_r.max().item()),
                "terminated=", int(terminated.sum().item()),
                "truncated=", int(truncated.sum().item()),
            )

        print("[TRACER] SMOKE TEST PASS")
        return 0

    except Exception:
        print("[TRACER] FAILED")
        traceback.print_exc()
        return 1

    finally:
        if env is not None:
            try:
                env.close()
                print("[TRACER] env closed")
            except Exception:
                print("[TRACER] env.close() failed")
                traceback.print_exc()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
