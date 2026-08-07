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


LEGS = ("FL", "FR", "RL", "RR")

OriginalQuadrupedEnv = sim.QuadrupedEnv


class LoggingQuadrupedEnv(OriginalQuadrupedEnv):
    feet_z = {leg: [] for leg in LEGS}

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)

        feet = super().feet_pos(frame="world")

        for leg in LEGS:
            p = np.asarray(feet[leg]).reshape(-1)
            type(self).feet_z[leg].append(float(p[2]))

        return result


def summarize(
    requested_clearance: float,
    output: Path,
) -> dict:

    dt = float(cfg.simulation_params["dt"])
    warmup_steps = int(2.0 / dt)

    per_leg = {}
    clearance_values = []

    for leg in LEGS:
        z = np.asarray(
            LoggingQuadrupedEnv.feet_z[leg],
            dtype=float,
        )

        z = z[warmup_steps:]

        if len(z) == 0:
            raise RuntimeError(
                f"No steady-state samples for {leg}"
            )

        # Robust stance-height estimate.
        stance_z = float(np.percentile(z, 5.0))

        # Robust swing-apex estimate.
        apex_z_p95 = float(np.percentile(z, 95.0))

        measured_clearance = (
            apex_z_p95 - stance_z
        )

        clearance_values.append(measured_clearance)

        per_leg[leg] = {
            "stance_z_p05_m": stance_z,
            "apex_z_p95_m": apex_z_p95,
            "max_z_m": float(np.max(z)),
            "clearance_p95_minus_p05_m":
                float(measured_clearance),
        }

    result = {
        "requested_swing_clearance_m":
            float(requested_clearance),

        "configured_step_height_m":
            float(cfg.simulation_params["step_height"]),

        "body_height_ref_m":
            float(cfg.simulation_params["ref_z"]),

        "gait":
            str(cfg.simulation_params["gait"]),

        "mean_measured_clearance_m":
            float(np.mean(clearance_values)),

        "std_across_legs_m":
            float(np.std(clearance_values)),

        "per_leg":
            per_leg,
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
            "ICRA27 PyMPC swing-clearance "
            "authority measurement"
        )
    )

    parser.add_argument(
        "--clearance",
        type=float,
        required=True,
        help="PyMPC step/swing height [m]",
    )

    parser.add_argument(
        "--vx",
        type=float,
        default=0.20,
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

    if not (0.015 <= args.clearance <= 0.12):
        raise ValueError(
            f"Refusing exploratory clearance: "
            f"{args.clearance:.3f} m"
        )

    LoggingQuadrupedEnv.feet_z = {
        leg: [] for leg in LEGS
    }

    sim.QuadrupedEnv = LoggingQuadrupedEnv

    # Fix all other main parameters.
    cfg.simulation_params["ref_z"] = 0.28

    # TRACER swing-clearance candidate
    # maps to PyMPC step_height.
    cfg.simulation_params["step_height"] = (
        float(args.clearance)
    )

    env_vx_argument = (
        float(args.vx) / float(cfg.hip_height)
    )

    gait = cfg.simulation_params["gait"]
    gait_params = (
        cfg.simulation_params["gait_params"][gait]
    )

    print("=" * 72)
    print("ICRA27 PyMPC swing-clearance authority")
    print("=" * 72)
    print(f"robot              : {cfg.robot}")
    print(f"gait               : {gait}")
    print(
        f"step frequency     : "
        f"{gait_params['step_freq']:.3f} Hz"
    )
    print(
        f"duty factor        : "
        f"{gait_params['duty_factor']:.3f}"
    )
    print(
        f"requested clearance: "
        f"{args.clearance:.3f} m"
    )
    print(
        f"configured height  : "
        f"{cfg.simulation_params['step_height']:.3f} m"
    )
    print(f"body height        : 0.280 m")
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
        requested_clearance=args.clearance,
        output=args.output,
    )

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print(
        "requested_swing_clearance_m :",
        result["requested_swing_clearance_m"],
    )
    print(
        "configured_step_height_m    :",
        result["configured_step_height_m"],
    )
    print(
        "mean_measured_clearance_m   :",
        result["mean_measured_clearance_m"],
    )
    print(
        "std_across_legs_m           :",
        result["std_across_legs_m"],
    )

    for leg, values in result["per_leg"].items():
        print(
            f"{leg}: clearance="
            f"{values['clearance_p95_minus_p05_m']:.6f} m"
        )

    print("=" * 72)
    print(
        "[ICRA27] swing-clearance measurement completed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
