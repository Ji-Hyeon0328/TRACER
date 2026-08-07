#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[2]
PYMPC_ROOT = ROOT / "external_baselines" / "Quadruped-PyMPC"
sys.path.insert(0, str(PYMPC_ROOT))

from quadruped_pympc import config as cfg
import simulation.simulation as sim


LEGS = ("FL", "FR", "RL", "RR")

OriginalQuadrupedEnv = sim.QuadrupedEnv


class LoggingQuadrupedEnv(OriginalQuadrupedEnv):
    feet_z = {leg: [] for leg in LEGS}
    base_vx = []

    def base_lin_vel(self, *args, **kwargs):
        value = super().base_lin_vel(*args, **kwargs)

        arr = np.asarray(value).reshape(-1)
        if arr.size:
            type(self).base_vx.append(float(arr[0]))

        return value

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)

        feet = super().feet_pos(frame="world")

        for leg in LEGS:
            p = np.asarray(feet[leg]).reshape(-1)
            type(self).feet_z[leg].append(float(p[2]))

        return result


def estimate_frequency(z, dt, commanded_freq):
    z = np.asarray(z, dtype=float)

    # Remove first two seconds.
    warmup = int(2.0 / dt)
    z = z[warmup:]

    if len(z) < 10:
        raise RuntimeError("Insufficient foot-height samples.")

    # Foot swing produces a clear local maximum.
    # Require peaks to be separated by at least ~45% of
    # the commanded gait period.
    min_distance = max(
        1,
        int(0.45 / (commanded_freq * dt))
    )

    prominence = max(
        0.005,
        0.20 * (np.percentile(z, 95) - np.percentile(z, 5))
    )

    peaks, _ = find_peaks(
        z,
        distance=min_distance,
        prominence=prominence,
    )

    if len(peaks) < 2:
        return {
            "num_peaks": int(len(peaks)),
            "measured_frequency_hz": None,
            "mean_period_s": None,
            "std_period_s": None,
        }

    periods = np.diff(peaks) * dt

    return {
        "num_peaks": int(len(peaks)),
        "measured_frequency_hz":
            float(1.0 / np.mean(periods)),
        "mean_period_s":
            float(np.mean(periods)),
        "std_period_s":
            float(np.std(periods)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frequency",
        type=float,
        required=True,
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

    if not (0.7 <= args.frequency <= 2.5):
        raise ValueError(
            f"Refusing exploratory frequency: "
            f"{args.frequency:.3f} Hz"
        )

    LoggingQuadrupedEnv.feet_z = {
        leg: [] for leg in LEGS
    }
    LoggingQuadrupedEnv.base_vx = []

    sim.QuadrupedEnv = LoggingQuadrupedEnv

    gait = "trot"

    # Fix everything except gait frequency.
    cfg.simulation_params["gait"] = gait
    cfg.simulation_params["ref_z"] = 0.28
    cfg.simulation_params["step_height"] = 0.06

    gait_params = cfg.simulation_params["gait_params"][gait]

    gait_params["step_freq"] = float(args.frequency)
    gait_params["duty_factor"] = 0.65

    # Prevent the controller from replacing our fixed command.
    cfg.mpc_params["optimize_step_freq"] = False

    env_vx_argument = (
        float(args.vx) / float(cfg.hip_height)
    )

    commanded_period = 1.0 / args.frequency
    stance_time = 0.65 / args.frequency
    swing_period = 0.35 / args.frequency

    print("=" * 72)
    print("ICRA27 PyMPC gait-frequency authority")
    print("=" * 72)
    print(f"gait                 : {gait}")
    print(f"commanded frequency  : {args.frequency:.3f} Hz")
    print(f"commanded period     : {commanded_period:.3f} s")
    print(f"expected stance time : {stance_time:.3f} s")
    print(f"expected swing period: {swing_period:.3f} s")
    print(f"duty factor          : 0.650")
    print(f"fixed vx             : {args.vx:.3f} m/s")
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

    dt = float(cfg.simulation_params["dt"])

    per_leg = {
        leg: estimate_frequency(
            LoggingQuadrupedEnv.feet_z[leg],
            dt,
            args.frequency,
        )
        for leg in LEGS
    }

    valid_freq = [
        d["measured_frequency_hz"]
        for d in per_leg.values()
        if d["measured_frequency_hz"] is not None
    ]

    vx = np.asarray(
        LoggingQuadrupedEnv.base_vx,
        dtype=float,
    )

    warmup = int(2.0 / dt)
    vx_ss = vx[warmup:]

    result = {
        "commanded_frequency_hz":
            float(args.frequency),

        "commanded_period_s":
            float(commanded_period),

        "configured_step_frequency_hz":
            float(gait_params["step_freq"]),

        "configured_duty_factor":
            float(gait_params["duty_factor"]),

        "expected_stance_time_s":
            float(stance_time),

        "expected_swing_period_s":
            float(swing_period),

        "mean_measured_frequency_hz":
            float(np.mean(valid_freq))
            if valid_freq else None,

        "std_across_legs_hz":
            float(np.std(valid_freq))
            if valid_freq else None,

        "mean_measured_vx_mps":
            float(np.mean(vx_ss))
            if len(vx_ss) else None,

        "per_leg":
            per_leg,
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

    for key in (
        "commanded_frequency_hz",
        "configured_step_frequency_hz",
        "mean_measured_frequency_hz",
        "std_across_legs_hz",
        "mean_measured_vx_mps",
    ):
        print(f"{key:34s}: {result[key]}")

    for leg in LEGS:
        print(
            f"{leg}: "
            f"{per_leg[leg]['measured_frequency_hz']} Hz "
            f"({per_leg[leg]['num_peaks']} peaks)"
        )

    print("=" * 72)
    print(
        "[ICRA27] gait-frequency measurement completed."
    )


if __name__ == "__main__":
    main()
