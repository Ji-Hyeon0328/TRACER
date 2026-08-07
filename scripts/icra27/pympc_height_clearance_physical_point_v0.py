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

import pympc_frequency_duty_physical_point_v0 as fd

from gym_quadruped.quadruped_env import QuadrupedEnv
from tracer_core.highlevel.meta_gait import MetaGaitCommand


LEGS = tuple(fd.LEGS)

ORIGINAL_COMPUTE_ACTIONS = (
    fd.base.sim.QuadrupedPyMPC_Wrapper.compute_actions
)

# Patch the exact environment class used by upstream
# run_simulation(), rather than assuming that our locally
# imported QuadrupedEnv object is the same runtime class.
SIM_GLOBALS = fd.base.sim.run_simulation.__globals__
SIM_ENV_CLASS = SIM_GLOBALS.get("QuadrupedEnv")

if SIM_ENV_CLASS is None:
    candidates = sorted(
        name
        for name, value in SIM_GLOBALS.items()
        if isinstance(value, type)
        and hasattr(value, "step")
    )

    raise RuntimeError(
        "Could not resolve QuadrupedEnv from "
        "run_simulation globals. "
        f"step-capable classes: {candidates}"
    )

ORIGINAL_ENV_STEP = SIM_ENV_CLASS.step


KIN_LOG = {
    "base_pos": [],
    "base_ori_rpy": [],
    "feet_pos_world": [],
}


def get_arg(args, kwargs, name, index):
    if name in kwargs:
        return kwargs[name]

    return args[index]


def leg_value(obj, leg, index):
    if hasattr(obj, leg):
        return getattr(obj, leg)

    try:
        return obj[leg]
    except (KeyError, TypeError, IndexError):
        pass

    return obj[index]


def leg_vector_array(obj):
    result = []

    for i, leg in enumerate(LEGS):
        value = np.asarray(
            leg_value(obj, leg, i),
            dtype=float,
        ).reshape(-1)

        if value.size < 3:
            padded = np.zeros(
                3,
                dtype=float,
            )
            padded[:value.size] = value
            value = padded

        result.append(
            value[:3]
        )

    return np.asarray(
        result,
        dtype=float,
    )


def traced_compute_actions(
    self,
    *args,
    **kwargs,
):
    base_pos = np.asarray(
        get_arg(
            args,
            kwargs,
            "base_pos",
            1,
        ),
        dtype=float,
    ).reshape(-1)

    base_ori = np.asarray(
        get_arg(
            args,
            kwargs,
            "base_ori_euler_xyz",
            3,
        ),
        dtype=float,
    ).reshape(-1)

    feet_pos = leg_vector_array(
        get_arg(
            args,
            kwargs,
            "feet_pos",
            5,
        )
    )

    tau = fd.base.tracer_compute_actions(
        self,
        *args,
        **kwargs,
    )

    KIN_LOG["base_pos"].append(
        base_pos[:3].copy()
    )

    KIN_LOG["base_ori_rpy"].append(
        base_ori[:3].copy()
    )

    KIN_LOG["feet_pos_world"].append(
        feet_pos.copy()
    )

    # fd.instrumented_env_step captures physical contact/GRF
    # only when the matching planned-contact sample is ready.
    # This mirrors the validated frequency-duty runner.
    fd.CAPTURE_READY = True

    return tau


def clear_kin_log():
    for value in KIN_LOG.values():
        value.clear()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--body-height",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--swing-clearance",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    fd.clear_logs()
    clear_kin_log()

    command = MetaGaitCommand(
        vx=0.20,
        yaw_rate=0.0,

        body_height=(
            args.body_height
        ),

        swing_clearance=(
            args.swing_clearance
        ),

        gait_period=(
            1.0 / 1.4
        ),

        duty_factor=0.65,

        source_mode=(
            "icra27_m3_height_clearance"
        ),

        source_reason=(
            "M3 height-clearance "
            "authority/coupling characterization"
        ),
    )

    projected = fd.base.ADAPTER.project(
        command
    )

    fd.base.COMMAND = command

    fd.base.cfg.simulation_params[
        "gait"
    ] = "trot"

    fd.base.cfg.mpc_params[
        "optimize_step_freq"
    ] = False

    gait_cfg = (
        fd.base.cfg.simulation_params[
            "gait_params"
        ]["trot"]
    )

    # Hold timing at the nominal point.
    gait_cfg["step_freq"] = 1.4
    gait_cfg["duty_factor"] = 0.65

    # Direct-start initialization at the requested
    # height and clearance. This intentionally removes
    # structural-transition behavior from this slice.
    fd.base.cfg.simulation_params[
        "step_height"
    ] = float(
        projected.swing_clearance
    )

    fd.base.cfg.simulation_params[
        "ref_z"
    ] = float(
        projected.body_height
    )

    (
        fd.base.sim
        .QuadrupedPyMPC_Wrapper
        .compute_actions
    ) = traced_compute_actions

    # fd.instrumented_env_step writes into fd.PHYSICAL_LOG.
    # Make sure its delegate is also the exact runtime method.
    if hasattr(fd, "ORIGINAL_ENV_STEP"):
        fd.ORIGINAL_ENV_STEP = (
            ORIGINAL_ENV_STEP
        )

    SIM_ENV_CLASS.step = (
        fd.instrumented_env_step
    )

    print("=" * 72)
    print(
        "ICRA27 M3 BODY HEIGHT × "
        "SWING CLEARANCE"
    )
    print("=" * 72)

    print(
        "projected:"
        f" h={projected.body_height:.3f} m"
        f" clr={projected.swing_clearance:.3f} m"
        " f=1.400 Hz"
        " D=0.650"
        " vx=0.200 m/s"
    )

    print("=" * 72)

    try:
        fd.base.sim.run_simulation(
            qpympc_cfg=fd.base.cfg,
            num_episodes=1,

            num_seconds_per_episode=(
                args.duration
            ),

            ref_base_lin_vel=(
                0.20
                / float(
                    fd.base.cfg.hip_height
                )
            ),

            ref_base_ang_vel=0.0,
            friction_coeff=0.8,

            base_vel_command_type=(
                "forward"
            ),

            seed=args.seed,
            render=False,
            recording_path=None,
        )

    finally:
        SIM_ENV_CLASS.step = (
            ORIGINAL_ENV_STEP
        )

        (
            fd.base.sim
            .QuadrupedPyMPC_Wrapper
            .compute_actions
        ) = ORIGINAL_COMPUTE_ACTIONS

    # Reuse the already validated first-rollout-aware
    # physical-contact characterization.
    result = fd.summarize(
        command,
        projected,
        args.output,
    )

    base_pos = np.asarray(
        KIN_LOG["base_pos"],
        dtype=float,
    )

    base_ori = np.asarray(
        KIN_LOG["base_ori_rpy"],
        dtype=float,
    )

    feet_pos = np.asarray(
        KIN_LOG["feet_pos_world"],
        dtype=float,
    )

    n = min(
        len(base_pos),
        len(base_ori),
        len(feet_pos),
        len(fd.base.LOG["contacts"]),
        len(fd.PHYSICAL_LOG["contact"]),
    )

    if n <= 0:
        raise RuntimeError(
            "No aligned kinematic samples"
        )

    dt = float(
        fd.base.cfg.simulation_params[
            "dt"
        ]
    )

    warmup = int(
        2.0 / dt
    )

    if fd.TERMINATION_EVENTS:
        first_termination_index = min(
            int(
                event[
                    "global_sample_index"
                ]
            )
            for event
            in fd.TERMINATION_EVENTS
        )

        analysis_stop = min(
            first_termination_index,
            n,
        )
    else:
        first_termination_index = None
        analysis_stop = n

    steady_state_available = (
        analysis_stop > warmup
    )

    if steady_state_available:
        analysis_start = warmup
        analysis_window_kind = (
            "steady_state"
        )
    else:
        analysis_start = 0
        analysis_window_kind = (
            "pre_failure_diagnostic"
        )

    if analysis_stop <= analysis_start:
        raise RuntimeError(
            "No usable first-rollout samples"
        )

    sl = slice(
        analysis_start,
        analysis_stop,
    )

    z = base_pos[sl, 2]

    roll = base_ori[sl, 0]
    pitch = base_ori[sl, 1]

    y = base_pos[sl, 1]
    y0 = float(y[0])

    per_leg_clearance = {}

    for leg_i, leg in enumerate(
        LEGS
    ):
        foot_z = feet_pos[
            sl,
            leg_i,
            2,
        ]

        clearance = float(
            np.percentile(
                foot_z,
                95,
            )
            -
            np.percentile(
                foot_z,
                5,
            )
        )

        per_leg_clearance[
            leg
        ] = clearance

    physical_clearance = float(
        np.mean(
            list(
                per_leg_clearance.values()
            )
        )
    )

    measured_height = float(
        np.mean(z)
    )

    usable_tracking_point = (
        steady_state_available
        and not bool(
            first_termination_index
            is not None
        )
    )

    hc = {
        "command": {
            "body_height_m":
                float(
                    projected.body_height
                ),

            "swing_clearance_m":
                float(
                    projected.swing_clearance
                ),
        },

        "analysis": {
            "window_kind":
                analysis_window_kind,

            "start_index":
                int(
                    analysis_start
                ),

            "stop_index":
                int(
                    analysis_stop
                ),

            "sample_count":
                int(
                    analysis_stop
                    - analysis_start
                ),

            "duration_s":
                float(
                    (
                        analysis_stop
                        - analysis_start
                    )
                    * dt
                ),

            "first_termination_index":
                first_termination_index,

            "usable_tracking_point":
                bool(
                    usable_tracking_point
                ),
        },

        "measured": {
            "mean_base_height_m":
                measured_height,

            "body_height_error_m":
                float(
                    measured_height
                    - projected.body_height
                ),

            "mean_physical_clearance_m":
                physical_clearance,

            "clearance_error_m":
                float(
                    physical_clearance
                    - projected.swing_clearance
                ),

            "per_leg_clearance_m":
                per_leg_clearance,

            "max_abs_roll_deg":
                float(
                    np.degrees(
                        np.max(
                            np.abs(roll)
                        )
                    )
                ),

            "max_abs_pitch_deg":
                float(
                    np.degrees(
                        np.max(
                            np.abs(pitch)
                        )
                    )
                ),

            "min_base_height_m":
                float(
                    np.min(z)
                ),

            "max_abs_lateral_drift_m":
                float(
                    np.max(
                        np.abs(
                            y - y0
                        )
                    )
                ),
        },
    }

    result[
        "height_clearance_characterization"
    ] = hc

    args.output.write_text(
        json.dumps(
            result,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("HEIGHT / CLEARANCE RESULT")
    print("=" * 72)

    print(
        "analysis window:",
        analysis_window_kind,
    )

    print(
        "usable tracking point:",
        usable_tracking_point,
    )

    print(
        "body height:"
        f" cmd={projected.body_height:.4f}"
        f" measured={measured_height:.4f}"
        f" error="
        f"{measured_height - projected.body_height:+.4f}"
    )

    print(
        "clearance:"
        f" cmd={projected.swing_clearance:.4f}"
        f" measured={physical_clearance:.4f}"
        f" error="
        f"{physical_clearance - projected.swing_clearance:+.4f}"
    )

    print(
        "attitude:"
        f" roll_max="
        f"{hc['measured']['max_abs_roll_deg']:.2f} deg"
        f" pitch_max="
        f"{hc['measured']['max_abs_pitch_deg']:.2f} deg"
    )

    print(
        "base:"
        f" z_min="
        f"{hc['measured']['min_base_height_m']:.4f}"
        f" lateral_drift="
        f"{hc['measured']['max_abs_lateral_drift_m']:.4f}"
    )

    print(
        "saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
