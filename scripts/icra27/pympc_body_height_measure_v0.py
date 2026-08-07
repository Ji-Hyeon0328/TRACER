#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYMPC_ROOT = ROOT / "external_baselines" / "Quadruped-PyMPC"

sys.path.insert(0, str(PYMPC_ROOT))

from quadruped_pympc import config as cfg
import simulation.simulation as sim


OriginalQuadrupedEnv = sim.QuadrupedEnv


class LoggingQuadrupedEnv(OriginalQuadrupedEnv):
    base_z = []
    base_vx = []
    x_position = []

    def base_lin_vel(self, *args, **kwargs):
        value = super().base_lin_vel(*args, **kwargs)

        arr = np.asarray(value).reshape(-1)
        if arr.size:
            type(self).base_vx.append(float(arr[0]))

        return value

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)

        pos = np.asarray(self.base_pos).reshape(-1)

        if pos.size >= 3:
            type(self).x_position.append(float(pos[0]))
            type(self).base_z.append(float(pos[2]))

        return result


def summarize(
    requested_height: float,
    requested_vx: float,
    output: Path,
) -> dict:

    z = np.asarray(
        LoggingQuadrupedEnv.base_z,
        dtype=float,
    )

    vx = np.asarray(
        LoggingQuadrupedEnv.base_vx,
        dtype=float,
    )

    xpos = np.asarray(
        LoggingQuadrupedEnv.x_position,
        dtype=float,
    )

    dt = float(cfg.simulation_params["dt"])
    warmup_steps = int(2.0 / dt)

    z_ss = z[warmup_steps:]
    vx_ss = vx[warmup_steps:]

    if len(z_ss) == 0:
        raise RuntimeError(
            "No steady-state body-height samples recorded."
        )

    result = {
        "requested_body_height_m": float(requested_height),
        "configured_ref_z_m": float(
            cfg.simulation_params["ref_z"]
        ),
        "requested_vx_mps": float(requested_vx),
        "num_height_samples_total": int(len(z)),
        "num_height_samples_steady": int(len(z_ss)),
        "mean_measured_base_z_m": float(np.mean(z_ss)),
        "std_measured_base_z_m": float(np.std(z_ss)),
        "mean_abs_height_error_m": float(
            np.mean(np.abs(z_ss - requested_height))
        ),
        "mean_measured_vx_mps": (
            float(np.mean(vx_ss))
            if len(vx_ss)
            else None
        ),
        "x_displacement_m": (
            float(xpos[-1] - xpos[0])
            if len(xpos) >= 2
            else None
        ),
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "ICRA27 body-height authority test "
            "for Quadruped-PyMPC"
        )
    )

    parser.add_argument(
        "--height",
        type=float,
        required=True,
        help="Desired body/base height [m]",
    )

    parser.add_argument(
        "--vx",
        type=float,
        default=0.20,
        help="Fixed forward velocity [m/s]",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if not (0.18 <= args.height <= 0.38):
        raise ValueError(
            f"Refusing unsafe exploratory height: "
            f"{args.height:.3f} m"
        )

    LoggingQuadrupedEnv.base_z = []
    LoggingQuadrupedEnv.base_vx = []
    LoggingQuadrupedEnv.x_position = []

    sim.QuadrupedEnv = LoggingQuadrupedEnv

    # This is the actual height reference consumed by WBInterface.
    cfg.simulation_params["ref_z"] = float(args.height)

    env_vx_argument = (
        float(args.vx) / float(cfg.hip_height)
    )

    print("=" * 72)
    print("ICRA27 PyMPC body-height authority test")
    print("=" * 72)
    print(f"robot              : {cfg.robot}")
    print(f"nominal hip height : {cfg.hip_height:.3f} m")
    print(f"requested height   : {args.height:.3f} m")
    print(
        f"configured ref_z   : "
        f"{cfg.simulation_params['ref_z']:.3f} m"
    )
    print(f"fixed vx           : {args.vx:.3f} m/s")
    print(f"duration           : {args.duration:.1f} s")
    print(f"output             : {args.output}")
    print("=" * 72)

    sim.run_simulation(
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

    result = summarize(
        requested_height=args.height,
        requested_vx=args.vx,
        output=args.output,
    )

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    for key, value in result.items():
        print(f"{key:34s}: {value}")

    print("=" * 72)
    print(
        "[ICRA27] body-height measurement completed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
