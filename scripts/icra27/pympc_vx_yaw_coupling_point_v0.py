#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ICRA27_SCRIPTS = ROOT / "scripts" / "icra27"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ICRA27_SCRIPTS))

import pympc_meta_gait_integration_v0 as base

from tracer_core.highlevel.meta_gait import MetaGaitCommand


def clear_logs():
    for key, value in base.LOG.items():
        if key == "feet_z":
            for leg in base.LEGS:
                value[leg].clear()
        else:
            value.clear()

    base.ADAPTER.reset()
    base.COMMAND = None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vx",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--yaw-rate",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    clear_logs()

    command = MetaGaitCommand(
        vx=args.vx,
        yaw_rate=args.yaw_rate,

        # Keep the remaining four dimensions fixed
        # at the current M2 nominal operating point.
        body_height=0.30,
        swing_clearance=0.06,
        gait_period=1.0 / 1.4,
        duty_factor=0.65,

        source_mode="icra27_m3_vx_yaw",
        source_reason="M3 vx-yaw coupling point",
    )

    projected = base.ADAPTER.project(command)

    base.COMMAND = command

    base.cfg.simulation_params["gait"] = "trot"
    base.cfg.mpc_params["optimize_step_freq"] = False

    base.sim.QuadrupedPyMPC_Wrapper.compute_actions = (
        base.tracer_compute_actions
    )

    print("=" * 72)
    print("ICRA27 M3 vx × yaw coupling point")
    print("=" * 72)

    print(f"vx command       : {projected.vx:+.3f} m/s")
    print(
        f"yaw-rate command : "
        f"{projected.yaw_rate:+.3f} rad/s"
    )

    print("fixed:")
    print(
        f"  body height    : "
        f"{projected.body_height:.3f} m"
    )
    print(
        f"  clearance      : "
        f"{projected.swing_clearance:.3f} m"
    )
    print(
        f"  gait frequency : "
        f"{projected.gait_frequency:.3f} Hz"
    )
    print(
        f"  duty factor    : "
        f"{projected.duty_factor:.3f}"
    )

    print("=" * 72)

    base.sim.run_simulation(
        qpympc_cfg=base.cfg,
        num_episodes=1,
        num_seconds_per_episode=args.duration,

        # Upstream generator supplies heading direction;
        # integration hook normalizes magnitude to MetaGait vx.
        ref_base_lin_vel=(
            projected.vx
            / float(base.cfg.hip_height)
        ),

        ref_base_ang_vel=projected.yaw_rate,
        friction_coeff=0.8,
        base_vel_command_type="forward+rotate",
        seed=0,
        render=False,
        recording_path=None,
    )

    result = base.summarize(
        command,
        projected,
        args.output,
    )

    measured = result["measured"]

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    print(
        "body-frame forward vx : "
        f"{measured['mean_body_forward_vx_mps']:+.6f} m/s"
    )

    print(
        "measured yaw rate     : "
        f"{measured['mean_yaw_rate_rad_s']:+.6f} rad/s"
    )

    print(
        "yaw change            : "
        f"{measured['yaw_change_rad']:+.6f} rad"
    )

    print(
        "base z                : "
        f"{measured['mean_base_z_m']:.6f} m"
    )

    print(
        "base z std            : "
        f"{measured['std_base_z_m']:.6f} m"
    )

    print("=" * 72)
    print("[ICRA27] vx-yaw coupling point completed.")


if __name__ == "__main__":
    main()
