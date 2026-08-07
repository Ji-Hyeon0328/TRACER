#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ICRA27_SCRIPTS = ROOT / "scripts" / "icra27"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ICRA27_SCRIPTS))

import pympc_meta_gait_integration_v0 as base
import pympc_meta_gait_guarded_transition_v0 as guarded

from gym_quadruped.quadruped_env import QuadrupedEnv
from tracer_core.highlevel.meta_gait import MetaGaitCommand


ORIGINAL_ENV_STEP = QuadrupedEnv.step

TERMINATION_EVENTS = []


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(k): to_jsonable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_jsonable(v)
            for v in value
        ]

    return value


def instrumented_env_step(self, action):
    result = ORIGINAL_ENV_STEP(
        self,
        action=action,
    )

    (
        state,
        reward,
        is_terminated,
        is_truncated,
        info,
    ) = result

    if is_terminated or is_truncated:
        invalid_contacts = info.get(
            "invalid_contacts",
            {},
        )

        invalid_names = sorted(
            str(name)
            for name in invalid_contacts.keys()
        )

        out_of_bounds = bool(
            self._check_out_of_terrain_bounds()
        )

        if invalid_names:
            reason = (
                "invalid_nonfoot_ground_contact"
            )
        elif out_of_bounds:
            reason = "out_of_terrain_bounds"
        elif is_truncated:
            reason = "truncated"
        else:
            reason = "terminated_unknown"

        if base.LOG["step_freq"]:
            applied_frequency = float(
                base.LOG["step_freq"][-1]
            )
        else:
            applied_frequency = None

        if base.LOG["duty_factor"]:
            applied_duty = float(
                base.LOG["duty_factor"][-1]
            )
        else:
            applied_duty = None

        TERMINATION_EVENTS.append({
            "active_target_label":
                guarded.CURRENT_TARGET_LABEL,

            "env_step_num":
                int(self.step_num),

            "simulation_time_s":
                float(self.simulation_time),

            "reason":
                reason,

            "terminated":
                bool(is_terminated),

            "truncated":
                bool(is_truncated),

            "invalid_contact_names":
                invalid_names,

            "out_of_terrain_bounds":
                out_of_bounds,

            "base_position_m": [
                float(x)
                for x in self.base_pos
            ],

            "applied_frequency_hz":
                applied_frequency,

            "applied_duty_factor":
                applied_duty,
        })

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frequency",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--duty-factor",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--nominal-duration",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--target-duration",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Open the MuJoCo viewer.",
    )

    args = parser.parse_args()

    nominal = MetaGaitCommand(
        vx=0.20,
        yaw_rate=0.0,
        body_height=0.30,
        swing_clearance=0.06,
        gait_period=1.0 / 1.4,
        duty_factor=0.65,
        source_mode="icra27_guarded_ab",
        source_reason="nominal",
    )

    target = MetaGaitCommand(
        vx=0.20,
        yaw_rate=0.0,
        body_height=0.30,
        swing_clearance=0.06,
        gait_period=1.0 / args.frequency,
        duty_factor=args.duty_factor,
        source_mode="icra27_guarded_ab",
        source_reason="suspicious_target",
    )

    total_duration = (
        args.nominal_duration
        + args.target_duration
    )

    guarded.SCHEDULE = [
        (
            "nominal",
            0.0,
            args.nominal_duration,
            nominal,
        ),
        (
            "target",
            args.nominal_duration,
            total_duration,
            target,
        ),
    ]

    guarded.DURATION = total_duration

    guarded.clear_logs()
    TERMINATION_EVENTS.clear()

    # Explicitly initialize backend at the safe nominal gait.
    base.cfg.simulation_params["gait"] = "trot"
    base.cfg.mpc_params[
        "optimize_step_freq"
    ] = False

    gait_cfg = (
        base.cfg.simulation_params[
            "gait_params"
        ]["trot"]
    )

    gait_cfg["step_freq"] = 1.4
    gait_cfg["duty_factor"] = 0.65

    base.cfg.simulation_params[
        "step_height"
    ] = 0.06

    base.cfg.simulation_params[
        "ref_z"
    ] = 0.30

    base.sim.QuadrupedPyMPC_Wrapper.compute_actions = (
        guarded.guarded_compute_actions
    )

    QuadrupedEnv.step = instrumented_env_step

    print("=" * 72)
    print("ICRA27 M3 GUARDED FREQUENCY × DUTY DIAGNOSTIC")
    print("=" * 72)

    print(
        f"nominal : 0-{args.nominal_duration:.1f}s "
        "f=1.400 D=0.650"
    )

    print(
        f"target  : "
        f"{args.nominal_duration:.1f}-"
        f"{total_duration:.1f}s "
        f"f={args.frequency:.3f} "
        f"D={args.duty_factor:.3f}"
    )

    print("=" * 72)

    try:
        base.sim.run_simulation(
            qpympc_cfg=base.cfg,
            num_episodes=1,
            num_seconds_per_episode=total_duration,
            ref_base_lin_vel=(
                0.20
                / float(base.cfg.hip_height)
            ),
            ref_base_ang_vel=0.0,
            friction_coeff=0.8,
            base_vel_command_type="forward",
            seed=0,
            render=args.render,
            recording_path=None,
        )

    finally:
        QuadrupedEnv.step = ORIGINAL_ENV_STEP

    events = to_jsonable(
        guarded.STRUCTURAL_EVENTS
    )

    first_termination = (
        TERMINATION_EVENTS[0]
        if TERMINATION_EVENTS
        else None
    )

    if first_termination is None:
        classification = (
            "target_survived_guarded_transition"
        )

    elif (
        first_termination[
            "active_target_label"
        ] == "nominal"
    ):
        classification = (
            "nominal_failed_before_target"
        )

    else:
        classification = (
            "failure_after_target_request"
        )

    result = {
        "nominal": {
            "frequency_hz": 1.4,
            "duty_factor": 0.65,
            "duration_s":
                float(args.nominal_duration),
        },

        "target": {
            "frequency_hz":
                float(args.frequency),

            "duty_factor":
                float(args.duty_factor),

            "duration_s":
                float(args.target_duration),
        },

        "structural_events":
            events,

        "termination_events":
            to_jsonable(
                TERMINATION_EVENTS
            ),

        "first_termination":
            to_jsonable(
                first_termination
            ),

        "classification":
            classification,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("STRUCTURAL EVENTS")
    print("-" * 72)

    if not events:
        print("none")
    else:
        print(
            json.dumps(
                events,
                indent=2,
            )
        )

    print()
    print("FIRST TERMINATION")
    print("-" * 72)

    if first_termination is None:
        print("NONE")
    else:
        print(
            json.dumps(
                first_termination,
                indent=2,
            )
        )

    print()
    print(
        "classification:",
        classification,
    )

    print("saved:", args.output)


if __name__ == "__main__":
    main()
