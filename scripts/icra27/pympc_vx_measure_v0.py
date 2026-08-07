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
    measured_vx = []
    reference_vx = []
    x_position = []
    simulation_time_log = []

    def base_lin_vel(self, *args, **kwargs):
        value = super().base_lin_vel(*args, **kwargs)

        arr = np.asarray(value).reshape(-1)
        if arr.size:
            type(self).measured_vx.append(float(arr[0]))

        return value

    def target_base_vel(self, *args, **kwargs):
        lin_vel, ang_vel = super().target_base_vel(*args, **kwargs)

        arr = np.asarray(lin_vel).reshape(-1)
        if arr.size:
            type(self).reference_vx.append(float(arr[0]))

        return lin_vel, ang_vel

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)

        pos = np.asarray(self.base_pos).reshape(-1)
        if pos.size:
            type(self).x_position.append(float(pos[0]))

        type(self).simulation_time_log.append(
            float(self.simulation_time)
        )

        return result


def summarize(requested_vx: float, output: Path) -> dict:
    measured = np.asarray(
        LoggingQuadrupedEnv.measured_vx,
        dtype=float,
    )

    reference = np.asarray(
        LoggingQuadrupedEnv.reference_vx,
        dtype=float,
    )

    xpos = np.asarray(
        LoggingQuadrupedEnv.x_position,
        dtype=float,
    )

    # Ignore first two seconds to reduce initialization transient.
    warmup_steps = int(2.0 / cfg.simulation_params["dt"])

    measured_ss = measured[warmup_steps:]
    reference_ss = reference[warmup_steps:]

    n = min(len(measured_ss), len(reference_ss))

    if n == 0:
        raise RuntimeError("No usable velocity samples were recorded.")

    measured_ss = measured_ss[:n]
    reference_ss = reference_ss[:n]

    result = {
        "requested_vx_mps": float(requested_vx),
        "num_samples_total": int(len(measured)),
        "num_samples_steady": int(n),
        "mean_internal_ref_vx_mps": float(
            np.mean(reference_ss)
        ),
        "mean_measured_vx_mps": float(
            np.mean(measured_ss)
        ),
        "std_measured_vx_mps": float(
            np.std(measured_ss)
        ),
        "mean_abs_tracking_error_mps": float(
            np.mean(np.abs(reference_ss - measured_ss))
        ),
        "x_displacement_m": (
            float(xpos[-1] - xpos[0])
            if len(xpos) >= 2
            else None
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vx", type=float, required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Reset static logs.
    LoggingQuadrupedEnv.measured_vx = []
    LoggingQuadrupedEnv.reference_vx = []
    LoggingQuadrupedEnv.x_position = []
    LoggingQuadrupedEnv.simulation_time_log = []

    # Replace only the class used by upstream run_simulation().
    sim.QuadrupedEnv = LoggingQuadrupedEnv

    env_vx_argument = args.vx / cfg.hip_height

    print("=" * 72)
    print("ICRA27 quantitative vx authority test")
    print("=" * 72)
    print(f"requested vx : {args.vx:.3f} m/s")
    print(f"duration     : {args.duration:.1f} s")
    print(f"output       : {args.output}")
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

    result = summarize(args.vx, args.output)

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    for key, value in result.items():
        print(f"{key:32s}: {value}")

    print("=" * 72)
    print("[ICRA27] quantitative vx measurement completed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
