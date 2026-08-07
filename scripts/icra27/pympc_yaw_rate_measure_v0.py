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


def yaw_component(value) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)

    if arr.size == 0:
        raise RuntimeError("Empty angular velocity value")

    if arr.size >= 3:
        return float(arr[2])

    # Some command interfaces expose yaw rate as a scalar.
    return float(arr[-1])


class LoggingQuadrupedEnv(OriginalQuadrupedEnv):
    measured_yaw_rate = []
    reference_yaw_rate = []
    yaw_angle = []
    measured_vx = []

    def base_ang_vel(self, *args, **kwargs):
        value = super().base_ang_vel(*args, **kwargs)

        type(self).measured_yaw_rate.append(
            yaw_component(value)
        )

        return value

    def base_lin_vel(self, *args, **kwargs):
        value = super().base_lin_vel(*args, **kwargs)

        arr = np.asarray(value, dtype=float).reshape(-1)

        if arr.size:
            type(self).measured_vx.append(
                float(arr[0])
            )

        return value

    def target_base_vel(self, *args, **kwargs):
        lin_vel, ang_vel = super().target_base_vel(
            *args,
            **kwargs,
        )

        type(self).reference_yaw_rate.append(
            yaw_component(ang_vel)
        )

        return lin_vel, ang_vel

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)

        euler = np.asarray(
            self.base_ori_euler_xyz,
            dtype=float,
        ).reshape(-1)

        if euler.size >= 3:
            type(self).yaw_angle.append(
                float(euler[2])
            )

        return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "ICRA27 PyMPC yaw-rate authority measurement"
        )
    )

    parser.add_argument(
        "--yaw-rate",
        type=float,
        required=True,
        help="Desired yaw rate [rad/s]",
    )

    parser.add_argument(
        "--vx",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
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

    if not (-0.6 <= args.yaw_rate <= 0.6):
        raise ValueError(
            f"Refusing exploratory yaw rate: "
            f"{args.yaw_rate:.3f} rad/s"
        )

    LoggingQuadrupedEnv.measured_yaw_rate = []
    LoggingQuadrupedEnv.reference_yaw_rate = []
    LoggingQuadrupedEnv.yaw_angle = []
    LoggingQuadrupedEnv.measured_vx = []

    sim.QuadrupedEnv = LoggingQuadrupedEnv

    # Fix the other MetaGait dimensions.
    cfg.simulation_params["gait"] = "trot"
    cfg.simulation_params["ref_z"] = 0.28
    cfg.simulation_params["step_height"] = 0.06

    gait_params = (
        cfg.simulation_params["gait_params"]["trot"]
    )

    gait_params["step_freq"] = 1.4
    gait_params["duty_factor"] = 0.65

    cfg.mpc_params["optimize_step_freq"] = False

    env_vx_argument = (
        float(args.vx) / float(cfg.hip_height)
    )

    print("=" * 72)
    print("ICRA27 PyMPC yaw-rate authority")
    print("=" * 72)
    print(f"fixed vx            : {args.vx:.3f} m/s")
    print(
        f"commanded yaw rate  : "
        f"{args.yaw_rate:.3f} rad/s"
    )
    print("body height         : 0.280 m")
    print("clearance           : 0.060 m")
    print("frequency           : 1.400 Hz")
    print("duty factor         : 0.650")
    print(f"duration            : {args.duration:.1f} s")
    print("=" * 72)

    sim.run_simulation(
        qpympc_cfg=cfg,
        num_episodes=1,
        num_seconds_per_episode=args.duration,
        ref_base_lin_vel=env_vx_argument,
        ref_base_ang_vel=float(args.yaw_rate),
        friction_coeff=0.8,
        base_vel_command_type="forward+rotate",
        seed=0,
        render=not args.no_render,
        recording_path=None,
    )

    dt = float(cfg.simulation_params["dt"])
    warmup = int(2.0 / dt)

    measured = np.asarray(
        LoggingQuadrupedEnv.measured_yaw_rate,
        dtype=float,
    )[warmup:]

    reference = np.asarray(
        LoggingQuadrupedEnv.reference_yaw_rate,
        dtype=float,
    )[warmup:]

    vx = np.asarray(
        LoggingQuadrupedEnv.measured_vx,
        dtype=float,
    )[warmup:]

    yaw = np.asarray(
        LoggingQuadrupedEnv.yaw_angle,
        dtype=float,
    )

    if len(measured) == 0 or len(reference) == 0:
        raise RuntimeError(
            "No steady-state yaw-rate samples."
        )

    n = min(len(measured), len(reference))

    measured = measured[:n]
    reference = reference[:n]

    yaw_change = None

    if len(yaw) >= 2:
        yaw_unwrapped = np.unwrap(yaw)
        yaw_change = float(
            yaw_unwrapped[-1] - yaw_unwrapped[0]
        )

    result = {
        "commanded_yaw_rate_rad_s":
            float(args.yaw_rate),

        "mean_internal_ref_yaw_rate_rad_s":
            float(np.mean(reference)),

        "mean_measured_yaw_rate_rad_s":
            float(np.mean(measured)),

        "std_measured_yaw_rate_rad_s":
            float(np.std(measured)),

        "mean_abs_tracking_error_rad_s":
            float(
                np.mean(
                    np.abs(reference - measured)
                )
            ),

        "yaw_angle_change_rad":
            yaw_change,

        "mean_measured_vx_mps":
            float(np.mean(vx))
            if len(vx)
            else None,

        "num_samples_steady":
            int(n),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    for key, value in result.items():
        print(f"{key:38s}: {value}")

    print("=" * 72)
    print(
        "[ICRA27] yaw-rate measurement completed."
    )


if __name__ == "__main__":
    main()
