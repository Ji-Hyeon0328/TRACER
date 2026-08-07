#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ICRA27_SCRIPTS = ROOT / "scripts" / "icra27"
PYMPC_ROOT = ROOT / "external_baselines" / "Quadruped-PyMPC"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ICRA27_SCRIPTS))
sys.path.insert(0, str(PYMPC_ROOT))

import pympc_meta_gait_integration_v0 as base

from gym_quadruped.quadruped_env import QuadrupedEnv

from tracer_core.highlevel.meta_gait import MetaGaitCommand


LEGS = ("FL", "FR", "RL", "RR")

ORIGINAL_ENV_STEP = QuadrupedEnv.step

CAPTURE_READY = False

PHYSICAL_LOG = {
    "contact": [],
    "grf_world": [],
}

TERMINATION_EVENTS = []


def clear_logs():
    global CAPTURE_READY

    for key, value in base.LOG.items():
        if key == "feet_z":
            for leg in base.LEGS:
                value[leg].clear()
        else:
            value.clear()

    base.ADAPTER.reset()
    base.COMMAND = None

    PHYSICAL_LOG["contact"].clear()
    PHYSICAL_LOG["grf_world"].clear()
    TERMINATION_EVENTS.clear()

    CAPTURE_READY = False


def instrumented_compute_actions(
    self,
    *args,
    **kwargs,
):
    global CAPTURE_READY

    tau = base.tracer_compute_actions(
        self,
        *args,
        **kwargs,
    )

    # planned contact has now been written to base.LOG.
    # The immediately following env.step() may capture the
    # corresponding physical MuJoCo contact state.
    CAPTURE_READY = True

    return tau


def instrumented_env_step(
    self,
    action,
):
    global CAPTURE_READY

    if CAPTURE_READY:
        contact_state, _, grf_world = (
            self.feet_contact_state(
                frame="world",
                ground_reaction_forces=True,
            )
        )

        PHYSICAL_LOG["contact"].append(
            [
                bool(contact_state[leg])
                for leg in LEGS
            ]
        )

        PHYSICAL_LOG["grf_world"].append(
            [
                np.asarray(
                    grf_world[leg],
                    dtype=float,
                ).reshape(3).tolist()
                for leg in LEGS
            ]
        )

        CAPTURE_READY = False

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

        invalid_contact_names = sorted(
            str(name)
            for name in invalid_contacts.keys()
        )

        out_of_bounds = bool(
            self._check_out_of_terrain_bounds()
        )

        if invalid_contact_names:
            reason = (
                "invalid_nonfoot_ground_contact"
            )
        elif out_of_bounds:
            reason = "out_of_terrain_bounds"
        elif is_truncated:
            reason = "truncated"
        else:
            reason = "terminated_unknown"

        TERMINATION_EVENTS.append({
            "global_sample_index":
                int(
                    len(
                        PHYSICAL_LOG["contact"]
                    ) - 1
                ),

            "env_step_num":
                int(self.step_num),

            "simulation_time_s":
                float(self.simulation_time),

            "terminated":
                bool(is_terminated),

            "truncated":
                bool(is_truncated),

            "reason":
                reason,

            "invalid_contact_names":
                invalid_contact_names,

            "out_of_terrain_bounds":
                out_of_bounds,

            "base_position_m": [
                float(x)
                for x in self.base_pos
            ],
        })

    return result


def summarize(
    command,
    projected,
    output,
):
    planned = np.asarray(
        base.LOG["contacts"],
        dtype=float,
    )

    physical = np.asarray(
        PHYSICAL_LOG["contact"],
        dtype=float,
    )

    grf = np.asarray(
        PHYSICAL_LOG["grf_world"],
        dtype=float,
    )

    print()
    print("sample counts:")
    print("  planned :", len(planned))
    print("  physical:", len(physical))
    print("  GRF     :", len(grf))

    if len(planned) != len(physical):
        raise RuntimeError(
            "planned/physical contact sample-count mismatch: "
            f"{len(planned)} != {len(physical)}"
        )

    if len(physical) != len(grf):
        raise RuntimeError(
            "physical-contact/GRF sample-count mismatch"
        )

    dt = float(base.cfg.simulation_params["dt"])

    # Characterization must only use the first continuous
    # rollout. Data after an upstream automatic reset must
    # never contaminate tracking statistics.
    warmup = int(2.0 / dt)

    if TERMINATION_EVENTS:
        first_termination_index = min(
            event["global_sample_index"]
            for event in TERMINATION_EVENTS
        )
        analysis_stop = first_termination_index
    else:
        first_termination_index = None
        analysis_stop = len(planned)

    analysis_start = warmup

    steady_state_available = (
        analysis_stop > analysis_start
    )

    if steady_state_available:
        analysis_window_start = analysis_start
        analysis_window_kind = "steady_state"
    else:
        # An early failure is itself useful feasibility
        # evidence. Preserve pre-failure samples for
        # diagnostics, but never treat them as a valid
        # steady-state tracking point.
        analysis_window_start = 0
        analysis_window_kind = (
            "pre_failure_diagnostic"
        )

    if analysis_stop <= analysis_window_start:
        raise RuntimeError(
            "No first-rollout samples available: "
            f"start={analysis_window_start}, "
            f"stop={analysis_stop}"
        )

    planned = planned[
        analysis_window_start:analysis_stop
    ]
    physical = physical[
        analysis_window_start:analysis_stop
    ]
    grf = grf[
        analysis_window_start:analysis_stop
    ]

    planned_per_leg = np.mean(
        planned,
        axis=0,
    )

    physical_per_leg = np.mean(
        physical,
        axis=0,
    )

    duty_error_per_leg = (
        physical_per_leg
        - planned_per_leg
    )

    agreement_per_leg = np.mean(
        planned == physical,
        axis=0,
    )

    # World-frame vertical GRF. Average over all steady samples.
    mean_grf_z_per_leg = np.mean(
        grf[:, :, 2],
        axis=0,
    )

    # Average vertical force only during physical contact.
    contact_grf_z_per_leg = []

    for i in range(4):
        mask = physical[:, i] > 0.5

        if np.any(mask):
            value = float(
                np.mean(
                    grf[mask, i, 2]
                )
            )
        else:
            value = None

        contact_grf_z_per_leg.append(value)

    result = {
        "command": {
            "vx_mps":
                float(command.vx),

            "yaw_rate_rad_s":
                float(command.yaw_rate),

            "body_height_m":
                float(command.body_height),

            "swing_clearance_m":
                float(command.swing_clearance),

            "gait_frequency_hz":
                float(projected.gait_frequency),

            "duty_factor":
                float(projected.duty_factor),
        },

        "samples": {
            "dt_s": dt,

            "warmup_s": 2.0,

            "num_total":
                int(len(base.LOG["contacts"])),

            "steady_state_start_index":
                int(analysis_start),

            "analysis_start_index":
                int(analysis_window_start),

            "analysis_stop_index":
                int(analysis_stop),

            "analysis_window_kind":
                analysis_window_kind,

            "steady_state_available":
                bool(steady_state_available),

            "first_termination_index":
                (
                    int(first_termination_index)
                    if first_termination_index
                    is not None
                    else None
                ),

            "num_steady":
                int(len(planned)),

            "analysis_duration_s":
                float(len(planned) * dt),
        },

        "per_leg": {
            leg: {
                "planned_duty_factor":
                    float(
                        planned_per_leg[i]
                    ),

                "physical_duty_factor":
                    float(
                        physical_per_leg[i]
                    ),

                "physical_minus_planned":
                    float(
                        duty_error_per_leg[i]
                    ),

                "contact_agreement_ratio":
                    float(
                        agreement_per_leg[i]
                    ),

                "mean_world_grf_z_n":
                    float(
                        mean_grf_z_per_leg[i]
                    ),

                "mean_contact_world_grf_z_n":
                    contact_grf_z_per_leg[i],
            }
            for i, leg in enumerate(LEGS)
        },

        "stability": {
            "num_terminations":
                int(
                    sum(
                        event["terminated"]
                        for event
                        in TERMINATION_EVENTS
                    )
                ),

            "num_truncations":
                int(
                    sum(
                        event["truncated"]
                        for event
                        in TERMINATION_EVENTS
                    )
                ),

            "valid_continuous_episode":
                bool(
                    len(
                        TERMINATION_EVENTS
                    ) == 0
                ),

            "usable_tracking_point":
                bool(
                    len(
                        TERMINATION_EVENTS
                    ) == 0
                    and steady_state_available
                ),

            "first_termination":
                (
                    TERMINATION_EVENTS[0]
                    if TERMINATION_EVENTS
                    else None
                ),

            "events":
                TERMINATION_EVENTS,
        },

        "summary": {
            "mean_planned_duty_factor":
                float(
                    np.mean(
                        planned_per_leg
                    )
                ),

            "mean_physical_duty_factor":
                float(
                    np.mean(
                        physical_per_leg
                    )
                ),

            "mean_physical_minus_planned":
                float(
                    np.mean(
                        duty_error_per_leg
                    )
                ),

            "mean_contact_agreement_ratio":
                float(
                    np.mean(
                        agreement_per_leg
                    )
                ),

            "max_abs_leg_duty_error":
                float(
                    np.max(
                        np.abs(
                            duty_error_per_leg
                        )
                    )
                ),
        },
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frequency",
        type=float,
        default=1.4,
    )

    parser.add_argument(
        "--duty-factor",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
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

    clear_logs()

    command = MetaGaitCommand(
        vx=0.20,
        yaw_rate=0.0,
        body_height=0.30,
        swing_clearance=0.06,

        gait_period=(
            1.0 / args.frequency
        ),

        duty_factor=args.duty_factor,

        source_mode="icra27_m3_frequency_duty",
        source_reason=(
            "M3 planned-vs-physical contact characterization"
        ),
    )

    projected = base.ADAPTER.project(
        command
    )

    base.COMMAND = command

    base.cfg.simulation_params["gait"] = "trot"
    base.cfg.mpc_params["optimize_step_freq"] = False

    # IMPORTANT:
    # Initialize the PyMPC backend directly at the requested
    # structural gait parameters. Otherwise WBInterface starts
    # from the upstream trot defaults (1.4 Hz, D=0.65) and the
    # adapter performs an abrupt structural change at the first
    # control tick. That would mix startup-transition robustness
    # with steady gait feasibility.
    gait_cfg = (
        base.cfg.simulation_params[
            "gait_params"
        ]["trot"]
    )

    gait_cfg["step_freq"] = float(
        projected.gait_frequency
    )

    gait_cfg["duty_factor"] = float(
        projected.duty_factor
    )

    # Also remove avoidable startup differences in the other
    # structural/reference channels.
    base.cfg.simulation_params[
        "step_height"
    ] = float(
        projected.swing_clearance
    )

    base.cfg.simulation_params[
        "ref_z"
    ] = float(
        projected.body_height
    )

    print("backend initialization:")
    print(
        "  cfg frequency :",
        gait_cfg["step_freq"],
    )
    print(
        "  cfg duty      :",
        gait_cfg["duty_factor"],
    )
    print(
        "  cfg clearance :",
        base.cfg.simulation_params[
            "step_height"
        ],
    )
    print(
        "  cfg ref_z     :",
        base.cfg.simulation_params[
            "ref_z"
        ],
    )

    base.sim.QuadrupedPyMPC_Wrapper.compute_actions = (
        instrumented_compute_actions
    )

    QuadrupedEnv.step = (
        instrumented_env_step
    )

    print("=" * 72)
    print(
        "ICRA27 M3 planned vs physical contact"
    )
    print("=" * 72)

    print(
        f"frequency        : "
        f"{projected.gait_frequency:.3f} Hz"
    )

    print(
        f"duty command     : "
        f"{projected.duty_factor:.3f}"
    )

    print("fixed:")
    print("  vx             : 0.200 m/s")
    print("  yaw            : 0.000 rad/s")
    print("  body height    : 0.300 m")
    print("  clearance      : 0.060 m")

    print("=" * 72)

    try:
        base.sim.run_simulation(
            qpympc_cfg=base.cfg,
            num_episodes=1,
            num_seconds_per_episode=(
                args.duration
            ),
            ref_base_lin_vel=(
                projected.vx
                / float(base.cfg.hip_height)
            ),
            ref_base_ang_vel=0.0,
            friction_coeff=0.8,
            base_vel_command_type="forward",
            seed=args.seed,
            render=False,
            recording_path=None,
        )

    finally:
        # Do not leak monkey patches if this module is imported
        # or reused in-process.
        QuadrupedEnv.step = ORIGINAL_ENV_STEP

    result = summarize(
        command,
        projected,
        args.output,
    )

    print()
    print("termination events:")

    if not TERMINATION_EVENTS:
        print("  none")
    else:
        for event in TERMINATION_EVENTS:
            print(
                "  "
                f"sample={event['global_sample_index']} "
                f"env_step={event['env_step_num']} "
                f"t={event['simulation_time_s']:.3f}s "
                f"reason={event['reason']} "
                f"contacts="
                f"{event['invalid_contact_names']} "
                f"base="
                f"{event['base_position_m']}"
            )

    print()
    print("=" * 72)
    print("PHYSICAL DUTY RESULT")
    print("=" * 72)

    for leg in LEGS:
        r = result["per_leg"][leg]

        print(
            f"{leg}: "
            f"planned={r['planned_duty_factor']:.6f} "
            f"physical={r['physical_duty_factor']:.6f} "
            f"error={r['physical_minus_planned']:+.6f} "
            f"agreement={r['contact_agreement_ratio']:.6f}"
        )

    s = result["summary"]

    print()
    print(
        "mean planned duty  : "
        f"{s['mean_planned_duty_factor']:.6f}"
    )

    print(
        "mean physical duty : "
        f"{s['mean_physical_duty_factor']:.6f}"
    )

    print(
        "physical - planned : "
        f"{s['mean_physical_minus_planned']:+.6f}"
    )

    print(
        "contact agreement   : "
        f"{s['mean_contact_agreement_ratio']:.6f}"
    )

    print(
        "max leg duty error  : "
        f"{s['max_abs_leg_duty_error']:.6f}"
    )

    print("=" * 72)
    print(
        "[ICRA27] physical contact characterization completed."
    )


if __name__ == "__main__":
    main()
