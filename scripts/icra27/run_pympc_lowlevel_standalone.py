#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYMPC_ROOT = (
    ROOT
    / "external_baselines"
    / "Quadruped-PyMPC"
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PYMPC_ROOT))


from gym_quadruped.quadruped_env import QuadrupedEnv

from quadruped_pympc import config as cfg
import simulation.simulation as sim

from tracer_core.highlevel.meta_gait import (
    MetaGaitCommand,
)
from tracer_core.lowlevel.pympc_lowlevel_runtime import (
    PyMPCLowLevelRuntime,
)


LEGS = ("FL", "FR", "RL", "RR")


NOMINAL = MetaGaitCommand(
    vx=0.20,
    yaw_rate=0.0,
    body_height=0.30,
    swing_clearance=0.06,
    gait_period=1.0 / 1.4,
    duty_factor=0.65,
    source_mode="icra27_standalone",
    source_reason="known nominal",
)


RUNTIME: PyMPCLowLevelRuntime | None = None
TARGET: MetaGaitCommand | None = None
ARGS = None

PENDING_SAMPLE: dict | None = None

LAST_REQUESTED_LABEL: str | None = None
LAST_SAFETY_STATE: str | None = None
LAST_COMMAND_SIGNATURE = None

STRUCTURAL_EVENTS: list[dict] = []
STATE_EVENTS: list[dict] = []
COMMAND_EVENTS: list[dict] = []
TERMINATION_EVENTS: list[dict] = []
RESET_EVENTS: list[dict] = []

FIRST_UNSAFE: dict | None = None
FIRST_OVERRIDE: dict | None = None
FIRST_BACKOFF_COMMIT: dict | None = None


ORIGINAL_COMPUTE_ACTIONS = (
    sim.QuadrupedPyMPC_Wrapper.compute_actions
)

ORIGINAL_ENV_STEP = QuadrupedEnv.step

ORIGINAL_WRAPPER_RESET = (
    sim.QuadrupedPyMPC_Wrapper.reset
)


def command_dict(command: Any) -> dict:
    return {
        "vx_mps": float(command.vx),
        "yaw_rate_rad_s": float(
            command.yaw_rate
        ),
        "body_height_m": float(
            command.body_height
        ),
        "swing_clearance_m": float(
            command.swing_clearance
        ),
        "gait_period_s": float(
            command.gait_period
        ),
        "gait_frequency_hz": float(
            1.0 / command.gait_period
        ),
        "duty_factor": float(
            command.duty_factor
        ),
    }


def leg_value(
    obj: Any,
    leg: str,
    index: int,
):
    if hasattr(obj, leg):
        return getattr(obj, leg)

    try:
        return obj[leg]
    except (KeyError, TypeError, IndexError):
        pass

    return obj[index]


def contact_array(obj: Any) -> np.ndarray:
    result = []

    for i, leg in enumerate(LEGS):
        value = leg_value(
            obj,
            leg,
            i,
        )

        result.append(
            bool(
                np.asarray(value)
                .reshape(-1)[0]
            )
        )

    return np.asarray(
        result,
        dtype=bool,
    )


def requested_for_time(
    time_s: float,
) -> tuple[str, MetaGaitCommand]:
    assert TARGET is not None
    assert ARGS is not None

    if time_s < ARGS.nominal_duration:
        return "nominal", NOMINAL

    return "target", TARGET


def make_world_velocity_reference(
    *,
    upstream_ref: Any,
    vx: float,
    base_ori_euler_xyz: Any,
) -> np.ndarray:
    """
    Preserve upstream world-frame heading while
    replacing only the XY speed magnitude.
    """

    ref_lin = np.asarray(
        upstream_ref,
        dtype=float,
    ).copy().reshape(-1)

    if ref_lin.size < 3:
        padded = np.zeros(
            3,
            dtype=float,
        )

        padded[:ref_lin.size] = ref_lin
        ref_lin = padded

    xy_norm = float(
        np.linalg.norm(ref_lin[:2])
    )

    if xy_norm > 1e-9:
        ref_lin[:2] *= (
            float(vx) / xy_norm
        )

    else:
        yaw = float(
            np.asarray(
                base_ori_euler_xyz,
                dtype=float,
            ).reshape(-1)[2]
        )

        ref_lin[0] = (
            float(vx) * np.cos(yaw)
        )

        ref_lin[1] = (
            float(vx) * np.sin(yaw)
        )

    ref_lin[2] = 0.0

    return ref_lin


def standalone_compute_actions(
    self,
    com_pos,
    base_pos,
    base_lin_vel,
    base_ori_euler_xyz,
    base_ang_vel,
    feet_pos,
    hip_pos,
    joints_pos,
    heightmaps,
    legs_order,
    simulation_dt,
    ref_base_lin_vel,
    ref_base_ang_vel,
    step_num,
    qpos,
    qvel,
    feet_jac,
    feet_jac_dot,
    feet_vel,
    legs_qfrc_passive,
    legs_qfrc_bias,
    legs_mass_matrix,
    legs_qpos_idx,
    legs_qvel_idx,
    tau,
    inertia,
    mujoco_contact,
):
    global PENDING_SAMPLE
    global LAST_REQUESTED_LABEL
    global LAST_COMMAND_SIGNATURE
    global FIRST_BACKOFF_COMMIT

    assert RUNTIME is not None

    dt = float(simulation_dt)
    time_s = float(step_num * dt)

    (
        requested_label,
        requested_command,
    ) = requested_for_time(time_s)

    resolution = RUNTIME.resolve_command(
        requested_label=requested_label,
        requested_command=requested_command,
        current_contact=(
            self.wb_interface.current_contact
        ),
        dt=dt,
    )

    if (
        requested_label
        != LAST_REQUESTED_LABEL
    ):
        print()
        print("=" * 72)
        print(
            "[ICRA27 standalone target] "
            f"t={time_s:.3f}s "
            f"-> {requested_label}"
        )

        projected = (
            resolution.requested_projected
        )

        print(
            "  requested: "
            f"vx={projected.vx:.3f} "
            f"h={projected.body_height:.3f} "
            f"clr={projected.swing_clearance:.3f} "
            f"f={projected.gait_frequency:.3f} "
            f"D={projected.duty_factor:.3f}"
        )
        print("=" * 72)

        LAST_REQUESTED_LABEL = (
            requested_label
        )

    applied = RUNTIME.apply_to_pympc(
        wb_interface=self.wb_interface,
        pympc_cfg=cfg,
    )

    if resolution.structural_commit:
        event = {
            "time_s": time_s,
            "requested_label":
                resolution.requested_label,
            "selected_label":
                resolution.selected_label,
            "planned_contact":
                contact_array(
                    self.wb_interface
                    .current_contact
                ).astype(int).tolist(),
            "applied":
                command_dict(applied),
        }

        STRUCTURAL_EVENTS.append(event)

        print(
            "[ICRA27 standalone structural commit] "
            f"t={time_s:.3f}s "
            f"selected="
            f"{resolution.selected_label} "
            f"f={applied.gait_frequency:.3f} "
            f"D={applied.duty_factor:.3f}"
        )

        if (
            resolution.selected_label
            == "m4_backoff_nominal"
            and FIRST_BACKOFF_COMMIT is None
        ):
            FIRST_BACKOFF_COMMIT = dict(
                event
            )

    signature = (
        resolution.requested_label,
        resolution.selected_label,
        bool(resolution.override_active),
        bool(resolution.structural_commit),
    )

    if signature != LAST_COMMAND_SIGNATURE:
        COMMAND_EVENTS.append({
            "time_s": time_s,
            "requested_label":
                resolution.requested_label,
            "selected_label":
                resolution.selected_label,
            "override_active":
                bool(
                    resolution.override_active
                ),
            "structural_commit":
                bool(
                    resolution.structural_commit
                ),
            "requested_projected":
                command_dict(
                    resolution
                    .requested_projected
                ),
            "selected_projected":
                command_dict(
                    resolution
                    .selected_projected
                ),
            "applied":
                command_dict(
                    resolution.applied
                ),
        })

        LAST_COMMAND_SIGNATURE = signature

    ref_lin = (
        make_world_velocity_reference(
            upstream_ref=ref_base_lin_vel,
            vx=applied.vx,
            base_ori_euler_xyz=(
                base_ori_euler_xyz
            ),
        )
    )

    ref_ang = np.zeros(
        3,
        dtype=float,
    )

    ref_ang[2] = applied.yaw_rate

    tau_out = ORIGINAL_COMPUTE_ACTIONS(
        self,
        com_pos,
        base_pos,
        base_lin_vel,
        base_ori_euler_xyz,
        base_ang_vel,
        feet_pos,
        hip_pos,
        joints_pos,
        heightmaps,
        legs_order,
        simulation_dt,
        ref_lin,
        ref_ang,
        step_num,
        qpos,
        qvel,
        feet_jac,
        feet_jac_dot,
        feet_vel,
        legs_qfrc_passive,
        legs_qfrc_bias,
        legs_mass_matrix,
        legs_qpos_idx,
        legs_qvel_idx,
        tau,
        inertia,
        mujoco_contact,
    )

    # Capture the same-tick planned state.
    #
    # Physical contact will be sampled at the
    # environment boundary immediately afterward,
    # preserving the already validated M4-v0
    # instrumentation semantics.
    base_pos_arr = np.asarray(
        base_pos,
        dtype=float,
    ).reshape(-1)

    base_ori_arr = np.asarray(
        base_ori_euler_xyz,
        dtype=float,
    ).reshape(-1)

    PENDING_SAMPLE = {
        "time_s": time_s,
        "dt": dt,

        "planned_contact":
            contact_array(
                self.wb_interface
                .current_contact
            ).copy(),

        "roll_rad":
            float(base_ori_arr[0]),

        "pitch_rad":
            float(base_ori_arr[1]),

        "base_height_m":
            float(base_pos_arr[2]),
    }

    return tau_out


def standalone_env_step(
    self,
    action,
):
    global PENDING_SAMPLE
    global LAST_SAFETY_STATE
    global FIRST_UNSAFE
    global FIRST_OVERRIDE

    assert RUNTIME is not None

    if PENDING_SAMPLE is not None:
        (
            physical_contact,
            _,
            _,
        ) = self.feet_contact_state(
            ground_reaction_forces=True
        )

        physical = contact_array(
            physical_contact
        )

        obs = RUNTIME.observe_physical_state(
            planned_contact=(
                PENDING_SAMPLE[
                    "planned_contact"
                ]
            ),
            physical_contact=physical,
            roll_rad=(
                PENDING_SAMPLE["roll_rad"]
            ),
            pitch_rad=(
                PENDING_SAMPLE["pitch_rad"]
            ),
            base_height_m=(
                PENDING_SAMPLE[
                    "base_height_m"
                ]
            ),
            dt=PENDING_SAMPLE["dt"],
            time_s=(
                PENDING_SAMPLE["time_s"]
            ),
        )

        state = obs.status.state.value

        event = {
            "time_s":
                float(
                    PENDING_SAMPLE[
                        "time_s"
                    ]
                ),

            "state":
                state,

            "support_deficit_fraction":
                float(
                    obs.status
                    .support_deficit_fraction
                ),

            "roll_deg":
                float(
                    np.degrees(
                        obs.status.roll_rad
                    )
                ),

            "pitch_deg":
                float(
                    np.degrees(
                        obs.status.pitch_rad
                    )
                ),

            "base_height_m":
                float(
                    obs.status.base_height_m
                ),

            "reasons":
                list(obs.status.reasons),
        }

        if state != LAST_SAFETY_STATE:
            STATE_EVENTS.append(event)

            print(
                "[M4 standalone] "
                f"t={event['time_s']:.3f}s "
                f"-> {state.upper():6s} "
                f"support="
                f"{event['support_deficit_fraction']:.3f} "
                f"roll="
                f"{event['roll_deg']:+.2f}deg "
                f"pitch="
                f"{event['pitch_deg']:+.2f}deg "
                f"z="
                f"{event['base_height_m']:.3f}"
            )

            LAST_SAFETY_STATE = state

        if (
            state == "unsafe"
            and FIRST_UNSAFE is None
        ):
            FIRST_UNSAFE = dict(event)

        if obs.override_activated:
            FIRST_OVERRIDE = {
                "time_s":
                    event["time_s"],
                "reasons":
                    list(
                        RUNTIME
                        .supervisor
                        .override_reasons
                    ),
            }

            print()
            print("=" * 72)
            print(
                "[M4 standalone ACTIVE] "
                "UNSAFE -> nominal back-off latched"
            )
            print(
                "  detection t : "
                f"{event['time_s']:.3f}s"
            )
            print(
                "  reasons     : "
                f"{FIRST_OVERRIDE['reasons']}"
            )
            print(
                "  NOTE: fallback still passes "
                "through TransitionManager."
            )
            print("=" * 72)

    result = ORIGINAL_ENV_STEP(
        self,
        action,
    )

    terminated = bool(result[2])
    truncated = bool(result[3])

    if terminated or truncated:
        event = {
            "time_s": (
                None
                if PENDING_SAMPLE is None
                else float(
                    PENDING_SAMPLE[
                        "time_s"
                    ]
                )
            ),
            "terminated": terminated,
            "truncated": truncated,
        }

        TERMINATION_EVENTS.append(
            event
        )

    PENDING_SAMPLE = None

    return result


def standalone_wrapper_reset(
    self,
    initial_feet_pos,
):
    global PENDING_SAMPLE
    global LAST_REQUESTED_LABEL
    global LAST_SAFETY_STATE
    global LAST_COMMAND_SIGNATURE

    assert RUNTIME is not None

    pre = {
        "monitor_state":
            RUNTIME
            .safety_monitor
            .last_status
            .state.value,

        "unsafe_latched":
            bool(
                RUNTIME
                .safety_monitor
                .unsafe_latched
            ),

        "override_active":
            bool(
                RUNTIME
                .supervisor
                .override_active
            ),

        "command_active":
            RUNTIME.command is not None,
    }

    result = ORIGINAL_WRAPPER_RESET(
        self,
        initial_feet_pos,
    )

    RUNTIME.reset_episode()

    PENDING_SAMPLE = None
    LAST_REQUESTED_LABEL = None
    LAST_SAFETY_STATE = None
    LAST_COMMAND_SIGNATURE = None

    post = {
        "monitor_state":
            RUNTIME
            .safety_monitor
            .last_status
            .state.value,

        "unsafe_latched":
            bool(
                RUNTIME
                .safety_monitor
                .unsafe_latched
            ),

        "override_active":
            bool(
                RUNTIME
                .supervisor
                .override_active
            ),

        "command_active":
            RUNTIME.command is not None,
    }

    RESET_EVENTS.append({
        "index":
            len(RESET_EVENTS),

        "pre":
            pre,

        "post":
            post,

        "lifecycle_reset_count":
            int(
                RUNTIME
                .lifecycle
                .reset_count
            ),
    })

    print()
    print(
        "[ICRA27 standalone reset] "
        "PyMPC -> TRACER runtime"
    )

    print(
        "  pre : "
        f"state={pre['monitor_state']} "
        f"unsafe={pre['unsafe_latched']} "
        f"override={pre['override_active']}"
    )

    print(
        "  post: "
        f"state={post['monitor_state']} "
        f"unsafe={post['unsafe_latched']} "
        f"override={post['override_active']}"
    )

    return result


def parse_args():
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
        "--vx",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--yaw-rate",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--body-height",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--clearance",
        type=float,
        default=0.06,
    )

    parser.add_argument(
        "--nominal-duration",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.frequency <= 0.0:
        parser.error(
            "--frequency must be > 0"
        )

    if args.duration <= 0.0:
        parser.error(
            "--duration must be > 0"
        )

    if (
        args.nominal_duration < 0.0
        or args.nominal_duration
        > args.duration
    ):
        parser.error(
            "--nominal-duration must satisfy "
            "0 <= nominal <= duration"
        )

    return args


def main():
    global RUNTIME
    global TARGET
    global ARGS

    ARGS = parse_args()

    TARGET = MetaGaitCommand(
        vx=ARGS.vx,
        yaw_rate=ARGS.yaw_rate,
        body_height=ARGS.body_height,
        swing_clearance=ARGS.clearance,
        gait_period=(
            1.0 / ARGS.frequency
        ),
        duty_factor=ARGS.duty_factor,
        source_mode="icra27_standalone",
        source_reason="CLI target",
    )

    RUNTIME = PyMPCLowLevelRuntime(
        fallback_command=NOMINAL,
    )

    # Match the validated M2/M3/M4 setup.
    cfg.simulation_params["gait"] = "trot"

    cfg.mpc_params[
        "optimize_step_freq"
    ] = False

    print("=" * 72)
    print(
        "ICRA27 CLOSED PyMPC LOW-LEVEL STANDALONE"
    )
    print("=" * 72)

    print(
        "nominal: "
        "vx=.200 h=.300 clr=.060 "
        "f=1.400 D=.650"
    )

    print(
        "target : "
        f"vx={ARGS.vx:.3f} "
        f"h={ARGS.body_height:.3f} "
        f"clr={ARGS.clearance:.3f} "
        f"f={ARGS.frequency:.3f} "
        f"D={ARGS.duty_factor:.3f}"
    )

    print(
        "schedule: "
        f"nominal={ARGS.nominal_duration:.3f}s "
        f"total={ARGS.duration:.3f}s"
    )

    print("=" * 72)

    hooks_restored = False

    sim.QuadrupedPyMPC_Wrapper.compute_actions = (
        standalone_compute_actions
    )

    QuadrupedEnv.step = (
        standalone_env_step
    )

    sim.QuadrupedPyMPC_Wrapper.reset = (
        standalone_wrapper_reset
    )

    try:
        sim.run_simulation(
            qpympc_cfg=cfg,
            num_episodes=1,
            num_seconds_per_episode=(
                ARGS.duration
            ),
            ref_base_lin_vel=(
                NOMINAL.vx
                / float(cfg.hip_height)
            ),
            ref_base_ang_vel=0.0,
            friction_coeff=0.8,
            base_vel_command_type="forward",
            seed=ARGS.seed,
            render=not ARGS.no_render,
            recording_path=None,
        )

    finally:
        (
            sim.QuadrupedPyMPC_Wrapper
            .compute_actions
        ) = ORIGINAL_COMPUTE_ACTIONS

        QuadrupedEnv.step = (
            ORIGINAL_ENV_STEP
        )

        (
            sim.QuadrupedPyMPC_Wrapper
            .reset
        ) = ORIGINAL_WRAPPER_RESET

        hooks_restored = (
            sim.QuadrupedPyMPC_Wrapper
            .compute_actions
            is ORIGINAL_COMPUTE_ACTIONS
            and QuadrupedEnv.step
            is ORIGINAL_ENV_STEP
            and sim.QuadrupedPyMPC_Wrapper
            .reset
            is ORIGINAL_WRAPPER_RESET
        )

    result = {
        "mode":
            "pympc_lowlevel_standalone_v0",

        "seed":
            int(ARGS.seed),

        "duration_s":
            float(ARGS.duration),

        "nominal_duration_s":
            float(
                ARGS.nominal_duration
            ),

        "nominal":
            command_dict(NOMINAL),

        "target":
            command_dict(TARGET),

        "first_unsafe":
            FIRST_UNSAFE,

        "first_override":
            FIRST_OVERRIDE,

        "first_backoff_commit":
            FIRST_BACKOFF_COMMIT,

        "termination_count":
            len(TERMINATION_EVENTS),

        "termination_events":
            TERMINATION_EVENTS,

        "reset_events":
            RESET_EVENTS,

        "structural_events":
            STRUCTURAL_EVENTS,

        "safety_state_events":
            STATE_EVENTS,

        "command_events":
            COMMAND_EVENTS,

        "hooks_restored":
            bool(hooks_restored),
    }

    if ARGS.output is not None:
        ARGS.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        ARGS.output.write_text(
            json.dumps(
                result,
                indent=2,
            )
            + "\n"
        )

    print()
    print("=" * 72)
    print("STANDALONE RESULT")
    print("=" * 72)

    print(
        "UNSAFE          :",
        (
            None
            if FIRST_UNSAFE is None
            else FIRST_UNSAFE["time_s"]
        ),
    )

    print(
        "override        :",
        FIRST_OVERRIDE is not None,
    )

    print(
        "backoff commit  :",
        (
            None
            if FIRST_BACKOFF_COMMIT
            is None
            else FIRST_BACKOFF_COMMIT[
                "time_s"
            ]
        ),
    )

    print(
        "terminations    :",
        len(TERMINATION_EVENTS),
    )

    print(
        "runtime resets  :",
        len(RESET_EVENTS),
    )

    print(
        "hooks restored  :",
        hooks_restored,
    )

    if ARGS.output is not None:
        print(
            "saved           :",
            ARGS.output,
        )

    print("=" * 72)

    if not hooks_restored:
        raise RuntimeError(
            "Standalone hooks were not restored"
        )


if __name__ == "__main__":
    main()
