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

# We patch only the Python method in memory.
# No upstream source file is modified.
OriginalComputeActions = (
    sim.QuadrupedPyMPC_Wrapper.compute_actions
)

CONTACT_LOG = []


def logging_compute_actions(self, *args, **kwargs):
    tau = OriginalComputeActions(
        self,
        *args,
        **kwargs,
    )

    CONTACT_LOG.append(
        np.asarray(
            self.wb_interface.current_contact,
            dtype=float,
        ).copy()
    )

    return tau


def main():
    parser = argparse.ArgumentParser(
        description=(
            "ICRA27 PyMPC duty-factor authority measurement"
        )
    )

    parser.add_argument(
        "--duty-factor",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--frequency",
        type=float,
        default=1.4,
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

    if not (0.45 <= args.duty_factor <= 0.85):
        raise ValueError(
            f"Refusing exploratory duty factor: "
            f"{args.duty_factor:.3f}"
        )

    if not (0.7 <= args.frequency <= 2.5):
        raise ValueError(
            f"Refusing exploratory frequency: "
            f"{args.frequency:.3f} Hz"
        )

    CONTACT_LOG.clear()

    # Runtime-only monkey patch.
    sim.QuadrupedPyMPC_Wrapper.compute_actions = (
        logging_compute_actions
    )

    gait = "trot"

    cfg.simulation_params["gait"] = gait
    cfg.simulation_params["ref_z"] = 0.28
    cfg.simulation_params["step_height"] = 0.06

    gait_params = (
        cfg.simulation_params["gait_params"][gait]
    )

    gait_params["step_freq"] = float(args.frequency)
    gait_params["duty_factor"] = float(args.duty_factor)

    cfg.mpc_params["optimize_step_freq"] = False

    env_vx_argument = (
        float(args.vx) / float(cfg.hip_height)
    )

    cycle_period = 1.0 / args.frequency

    expected_stance_time = (
        args.duty_factor * cycle_period
    )

    expected_swing_time = (
        (1.0 - args.duty_factor) * cycle_period
    )

    print("=" * 72)
    print("ICRA27 PyMPC duty-factor authority")
    print("=" * 72)
    print(f"gait                : {gait}")
    print(
        f"frequency           : "
        f"{args.frequency:.3f} Hz"
    )
    print(
        f"commanded duty      : "
        f"{args.duty_factor:.3f}"
    )
    print(
        f"cycle period        : "
        f"{cycle_period:.4f} s"
    )
    print(
        f"expected stance     : "
        f"{expected_stance_time:.4f} s"
    )
    print(
        f"expected swing      : "
        f"{expected_swing_time:.4f} s"
    )
    print(f"fixed vx            : {args.vx:.3f} m/s")
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

    contacts = np.asarray(
        CONTACT_LOG,
        dtype=float,
    )

    if contacts.ndim != 2 or contacts.shape[1] != 4:
        raise RuntimeError(
            f"Unexpected contact log shape: "
            f"{contacts.shape}"
        )

    dt = float(cfg.simulation_params["dt"])
    warmup = int(2.0 / dt)

    contacts_ss = contacts[warmup:]

    if len(contacts_ss) == 0:
        raise RuntimeError(
            "No steady-state contact samples."
        )

    # current_contact:
    #   1 = stance
    #   0 = swing
    measured_per_leg = np.mean(
        contacts_ss,
        axis=0,
    )

    result = {
        "commanded_duty_factor":
            float(args.duty_factor),

        "configured_duty_factor":
            float(gait_params["duty_factor"]),

        "configured_frequency_hz":
            float(gait_params["step_freq"]),

        "cycle_period_s":
            float(cycle_period),

        "expected_stance_time_s":
            float(expected_stance_time),

        "expected_swing_time_s":
            float(expected_swing_time),

        "mean_measured_duty_factor":
            float(np.mean(measured_per_leg)),

        "std_across_legs":
            float(np.std(measured_per_leg)),

        "per_leg": {
            leg: float(measured_per_leg[i])
            for i, leg in enumerate(LEGS)
        },

        "num_samples_total":
            int(len(contacts)),

        "num_samples_steady":
            int(len(contacts_ss)),
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

    print(
        "commanded_duty_factor        :",
        result["commanded_duty_factor"],
    )

    print(
        "configured_duty_factor       :",
        result["configured_duty_factor"],
    )

    print(
        "mean_measured_duty_factor    :",
        result["mean_measured_duty_factor"],
    )

    print(
        "std_across_legs              :",
        result["std_across_legs"],
    )

    for leg in LEGS:
        print(
            f"{leg}: duty="
            f"{result['per_leg'][leg]:.6f}"
        )

    print("=" * 72)
    print(
        "[ICRA27] duty-factor measurement completed."
    )


if __name__ == "__main__":
    main()
