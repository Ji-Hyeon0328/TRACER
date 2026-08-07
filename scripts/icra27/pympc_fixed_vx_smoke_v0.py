#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYMPC_ROOT = ROOT / "external_baselines" / "Quadruped-PyMPC"

if not PYMPC_ROOT.is_dir():
    raise RuntimeError(f"Quadruped-PyMPC not found: {PYMPC_ROOT}")

sys.path.insert(0, str(PYMPC_ROOT))

from quadruped_pympc import config as cfg
from simulation.simulation import run_simulation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ICRA27 fixed-vx authority smoke test for Quadruped-PyMPC"
    )
    parser.add_argument(
        "--vx",
        type=float,
        default=0.20,
        help="Desired forward velocity in m/s",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration in seconds",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
    )
    args = parser.parse_args()

    if cfg.hip_height <= 0.0:
        raise RuntimeError(f"Invalid hip_height: {cfg.hip_height}")

    # Quadruped-PyMPC simulation.py multiplies this argument by hip_height
    # before passing it to QuadrupedEnv.
    #
    # Therefore:
    #
    #   argument * hip_height = physical target vx [m/s]
    #
    env_vx_argument = args.vx / cfg.hip_height

    print("=" * 72)
    print("ICRA27 PyMPC fixed-vx smoke test")
    print("=" * 72)
    print(f"robot               : {cfg.robot}")
    print(f"hip_height          : {cfg.hip_height:.6f} m")
    print(f"requested vx        : {args.vx:.6f} m/s")
    print(f"run_simulation arg  : {env_vx_argument:.6f}")
    print(f"expected env target : {env_vx_argument * cfg.hip_height:.6f} m/s")
    print(f"duration            : {args.duration:.3f} s")
    print("=" * 72)

    run_simulation(
        qpympc_cfg=cfg,
        num_episodes=1,
        num_seconds_per_episode=args.duration,
        ref_base_lin_vel=env_vx_argument,
        ref_base_ang_vel=0.0,
        friction_coeff=0.8,
        base_vel_command_type="forward",
        seed=0,
        render=not args.no_render,
        recording_path=None,
    )

    print()
    print("[ICRA27] fixed-vx smoke completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
