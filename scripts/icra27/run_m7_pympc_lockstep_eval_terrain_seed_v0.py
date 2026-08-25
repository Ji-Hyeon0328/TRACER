#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
ICRA27 = ROOT / "scripts" / "icra27"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ICRA27))


import run_pympc_lowlevel_udp as m5
import run_m7_pympc_state_tap_v0 as state_tap
import run_m7_pympc_state_tap_terrain_seed_v0 as terrain_seed_runner


LOCKSTEP_PHYSICS_STEPS = int(
    os.environ.get(
        "TRACER_M7_LOCKSTEP_PHYSICS_STEPS",
        "100",
    )
)

LOCKSTEP_WAIT_TIMEOUT_S = float(
    os.environ.get(
        "TRACER_M7_LOCKSTEP_WAIT_TIMEOUT_S",
        "10.0",
    )
)

if LOCKSTEP_PHYSICS_STEPS <= 0:
    raise ValueError(
        "TRACER_M7_LOCKSTEP_PHYSICS_STEPS "
        "must be > 0"
    )


ORIGINAL_UDP_REQUESTED_FOR_TIME = (
    m5.udp_requested_for_time
)

ORIGINAL_STATE_READY = (
    state_tap.M7StateUdpSender.ready
)

ORIGINAL_STATE_TAP_COMPUTE_TARGET = (
    state_tap.ORIGINAL_STANDALONE_COMPUTE_ACTIONS
)

ORIGINAL_STATE_TAP_UDP_ENV_STEP = (
    state_tap.m7_udp_env_step
)


LATCHED_SEQ = None
LATCHED_COMMAND = None


def _packet_to_command(packet):
    return m5.MetaGaitCommand(
        vx=packet.vx,
        yaw_rate=packet.yaw_rate,
        body_height=packet.body_height,
        swing_clearance=(
            packet.swing_clearance
        ),
        gait_period=packet.gait_period,
        duty_factor=packet.duty_factor,
        source_mode="ros2_udp",
        source_reason=(
            f"lockstep UDP command seq="
            f"{packet.seq}"
        ),
    )


def _command_mapping(command):
    return {
        "vx": float(command.vx),
        "yaw_rate": float(
            command.yaw_rate
        ),
        "body_height": float(
            command.body_height
        ),
        "swing_clearance": float(
            command.swing_clearance
        ),
        "gait_period": float(
            command.gait_period
        ),
        "duty_factor": float(
            command.duty_factor
        ),
    }


def _wait_for_exact_seq(
    expected_seq: int,
):
    assert m5.INBOX is not None

    deadline = (
        time.monotonic()
        + LOCKSTEP_WAIT_TIMEOUT_S
    )

    while time.monotonic() < deadline:
        m5.INBOX.poll()

        packet = m5.INBOX.latest

        if packet is not None:
            seq = int(packet.seq)

            if seq > expected_seq:
                raise RuntimeError(
                    "Lockstep command sequence "
                    "skipped expected boundary: "
                    f"expected={expected_seq} "
                    f"received={seq}"
                )

            if seq == expected_seq:
                return (
                    packet,
                    _packet_to_command(
                        packet
                    ),
                )

        time.sleep(0.001)

    raise TimeoutError(
        "Timed out waiting for lockstep "
        f"command seq={expected_seq}"
    )


def lockstep_requested_for_time(
    time_s: float,
):
    global LATCHED_SEQ
    global LATCHED_COMMAND

    dt = float(
        m5.standalone
        .cfg
        .simulation_params["dt"]
    )

    tick_float = (
        float(time_s)
        / dt
    )

    tick = int(
        round(tick_float)
    )

    if abs(
        float(time_s)
        - float(tick) * dt
    ) > 1e-10:
        raise RuntimeError(
            "Simulation time is not on the "
            "physics grid: "
            f"time={time_s} dt={dt}"
        )

    m5.CURRENT_SIM_TIME_S = float(
        time_s
    )

    if (
        tick
        % LOCKSTEP_PHYSICS_STEPS
        == 0
    ):
        expected_seq = (
            0
            if LATCHED_SEQ is None
            else int(LATCHED_SEQ) + 1
        )

        packet, command = (
            _wait_for_exact_seq(
                expected_seq
            )
        )

        LATCHED_SEQ = int(
            packet.seq
        )

        LATCHED_COMMAND = command

    if (
        LATCHED_SEQ is None
        or LATCHED_COMMAND is None
    ):
        raise RuntimeError(
            "Lockstep simulator advanced "
            "without a latched command"
        )

    label = (
        f"ros2_seq_{LATCHED_SEQ}"
    )

    m5.CURRENT_REQUESTED_LABEL = (
        label
    )

    return (
        label,
        LATCHED_COMMAND,
    )


def never_async_ready(
    self,
):
    del self
    return False


def lockstep_state_boundary_bridge(
    self,
    *args,
    **kwargs,
):
    global LATCHED_SEQ

    if "simulation_dt" in kwargs:
        simulation_dt = float(
            kwargs["simulation_dt"]
        )
    else:
        simulation_dt = float(
            args[10]
        )

    if "step_num" in kwargs:
        step_num = int(
            kwargs["step_num"]
        )
    else:
        step_num = int(
            args[13]
        )

    if (
        step_num
        % LOCKSTEP_PHYSICS_STEPS
        == 0
    ):
        if state_tap.LATEST_STATE is None:
            raise RuntimeError(
                "Lockstep boundary reached "
                "without LATEST_STATE"
            )

        if state_tap.STATE_SENDER is None:
            raise RuntimeError(
                "Lockstep boundary reached "
                "without STATE_SENDER"
            )

        payload = dict(
            state_tap.LATEST_STATE
        )

        expected_time = (
            float(step_num)
            * simulation_dt
        )

        if abs(
            float(
                payload[
                    "sample_time_s"
                ]
            )
            - expected_time
        ) > 1e-12:
            raise RuntimeError(
                "Lockstep boundary state "
                "timestamp mismatch"
            )

        runtime = (
            m5.standalone.RUNTIME
        )

        if runtime is None:
            raise RuntimeError(
                "Lockstep boundary reached "
                "without low-level runtime"
            )

        applied = runtime.command

        if applied is None:
            applied = m5.NOMINAL

        status = (
            runtime
            .safety_monitor
            .last_status
        )

        safety_state = (
            "normal"
            if status is None
            else str(
                status.state.value
            )
        )

        payload[
            "terminated"
        ] = False

        payload[
            "truncated"
        ] = False

        payload[
            "mechanical_energy"
        ] = (
            state_tap
            .ENERGY_ACCUMULATOR
            .as_dict()
        )

        payload[
            "energy_sample_time_s"
        ] = float(
            payload[
                "sample_time_s"
            ]
        )

        payload[
            "eval_stance_slip"
        ] = (
            state_tap
            ._slip_payload()
        )

        payload[
            "lockstep_eval"
        ] = True

        payload[
            "lockstep_boundary"
        ] = True

        payload[
            "lockstep_step_num"
        ] = int(
            step_num
        )

        payload[
            "lockstep_physics_steps"
        ] = int(
            LOCKSTEP_PHYSICS_STEPS
        )

        payload[
            "lockstep_completed_command_seq"
        ] = (
            None
            if LATCHED_SEQ is None
            else int(
                LATCHED_SEQ
            )
        )

        payload[
            "lockstep_applied"
        ] = _command_mapping(
            applied
        )

        payload[
            "lockstep_safety_state"
        ] = safety_state

        payload[
            "lockstep_override_active"
        ] = bool(
            runtime
            .supervisor
            .override_active
        )

        payload[
            "lockstep_override_reasons"
        ] = list(
            runtime
            .supervisor
            .override_reasons
        )

        state_tap.STATE_SENDER.send(
            payload
        )

    return (
        ORIGINAL_STATE_TAP_COMPUTE_TARGET(
            self,
            *args,
            **kwargs,
        )
    )


def lockstep_m7_udp_env_step(
    self,
    action,
):
    """
    Preserve the existing M7 energy/slip/M5 step semantics,
    but emit an immediate lockstep boundary when native
    termination or an M4 execution boundary occurs.

    Physical state remains the validated pre-env-step sample.
    Post-step energy time supplies the effective terminal
    decision boundary.
    """

    result = (
        ORIGINAL_STATE_TAP_UDP_ENV_STEP(
            self,
            action,
        )
    )

    runtime = (
        m5.standalone.RUNTIME
    )

    if runtime is None:
        raise RuntimeError(
            "Lockstep env step completed "
            "without low-level runtime"
        )

    status = (
        runtime
        .safety_monitor
        .last_status
    )

    safety_state = (
        "normal"
        if status is None
        else str(
            status.state.value
        )
    )

    override_active = bool(
        runtime
        .supervisor
        .override_active
    )

    native_terminated = bool(
        result[2]
    )

    native_truncated = bool(
        result[3]
    )

    m4_interrupt = bool(
        safety_state.strip().lower()
        == "unsafe"
        or override_active
    )

    if not (
        native_terminated
        or native_truncated
        or m4_interrupt
    ):
        return result

    if state_tap.LATEST_STATE is None:
        raise RuntimeError(
            "Terminal lockstep event has "
            "no pre-step physical state"
        )

    if (
        state_tap
        .LATEST_ENERGY_SAMPLE
        is None
    ):
        raise RuntimeError(
            "Terminal lockstep event has "
            "no post-step energy sample"
        )

    if state_tap.STATE_SENDER is None:
        raise RuntimeError(
            "Terminal lockstep event has "
            "no state sender"
        )

    if LATCHED_SEQ is None:
        raise RuntimeError(
            "Terminal lockstep event occurred "
            "before any command was latched"
        )

    payload = dict(
        state_tap.LATEST_STATE
    )

    dt = float(
        payload[
            "lowlevel_dt_s"
        ]
    )

    physical_sample_time = float(
        payload[
            "sample_time_s"
        ]
    )

    physical_sample_step_num = int(
        round(
            physical_sample_time
            / dt
        )
    )

    if abs(
        physical_sample_time
        - float(
            physical_sample_step_num
        ) * dt
    ) > 1e-10:
        raise RuntimeError(
            "Terminal physical sample "
            "is off physics grid"
        )

    energy_sample = (
        state_tap
        .LATEST_ENERGY_SAMPLE
    )

    env_pre_time = float(
        energy_sample[
            "time_pre_s"
        ]
    )

    terminal_end_time = float(
        energy_sample[
            "time_post_s"
        ]
    )

    measured_step_dt = float(
        energy_sample[
            "dt_s"
        ]
    )

    if abs(
        measured_step_dt - dt
    ) > 1e-9:
        raise RuntimeError(
            "Terminal MuJoCo step dt "
            "does not match low-level dt: "
            f"measured={measured_step_dt:.15f} "
            f"expected={dt:.15f}"
        )

    if abs(
        (
            terminal_end_time
            - env_pre_time
        )
        - dt
    ) > 1e-9:
        raise RuntimeError(
            "Terminal env-step timing "
            "contract violated: "
            f"pre={env_pre_time:.15f} "
            f"post={terminal_end_time:.15f} "
            f"dt={dt:.15f}"
        )

    env_pre_step_num = int(
        round(
            env_pre_time
            / dt
        )
    )

    terminal_end_step_num = int(
        round(
            terminal_end_time
            / dt
        )
    )

    expected_pre_time = (
        float(
            env_pre_step_num
        )
        * dt
    )

    expected_end_time = (
        float(
            terminal_end_step_num
        )
        * dt
    )

    if abs(
        env_pre_time
        - expected_pre_time
    ) > 1e-9:
        raise RuntimeError(
            "Terminal env pre-step time "
            "is off physics grid: "
            f"actual={env_pre_time:.15f} "
            f"expected={expected_pre_time:.15f}"
        )

    if abs(
        terminal_end_time
        - expected_end_time
    ) > 1e-9:
        raise RuntimeError(
            "Terminal post-step time "
            "is off physics grid: "
            f"actual={terminal_end_time:.15f} "
            f"expected={expected_end_time:.15f}"
        )

    if (
        terminal_end_step_num
        - env_pre_step_num
        != 1
    ):
        raise RuntimeError(
            "Terminal env step did not "
            "advance exactly one physics tick"
        )

    physical_to_env_pre_lag_ticks = (
        env_pre_step_num
        - physical_sample_step_num
    )

    applied = runtime.command

    if applied is None:
        applied = m5.NOMINAL

    payload[
        "terminated"
    ] = native_terminated

    payload[
        "truncated"
    ] = native_truncated

    payload[
        "native_reward"
    ] = state_tap._native_reward(
        result
    )

    payload[
        "mechanical_energy"
    ] = dict(
        energy_sample[
            "cumulative_energy"
        ]
    )

    payload[
        "energy_sample_time_s"
    ] = terminal_end_time

    payload[
        "eval_stance_slip"
    ] = (
        state_tap
        ._slip_payload()
    )

    payload[
        "lockstep_eval"
    ] = True

    payload[
        "lockstep_boundary"
    ] = True

    payload[
        "lockstep_terminal"
    ] = True

    # Physical state remains the existing validated
    # controller-input sample.
    payload[
        "lockstep_step_num"
    ] = physical_sample_step_num

    payload[
        "lockstep_terminal_physical_sample_time_s"
    ] = physical_sample_time

    payload[
        "lockstep_terminal_env_pre_step_num"
    ] = env_pre_step_num

    payload[
        "lockstep_terminal_env_pre_time_s"
    ] = env_pre_time

    payload[
        "lockstep_terminal_physical_to_env_pre_lag_ticks"
    ] = physical_to_env_pre_lag_ticks

    # Authoritative logical interval endpoint:
    # actual MuJoCo post-step clock.
    payload[
        "lockstep_terminal_end_step_num"
    ] = terminal_end_step_num

    payload[
        "lockstep_terminal_end_time_s"
    ] = terminal_end_time

    payload[
        "lockstep_physics_steps"
    ] = int(
        LOCKSTEP_PHYSICS_STEPS
    )

    payload[
        "lockstep_completed_command_seq"
    ] = int(
        LATCHED_SEQ
    )

    payload[
        "lockstep_applied"
    ] = _command_mapping(
        applied
    )

    payload[
        "lockstep_safety_state"
    ] = safety_state

    payload[
        "lockstep_override_active"
    ] = override_active

    payload[
        "lockstep_override_reasons"
    ] = list(
        runtime
        .supervisor
        .override_reasons
    )

    payload[
        "lockstep_terminal_reason"
    ] = {
        "native_terminated":
            native_terminated,
        "native_truncated":
            native_truncated,
        "m4_interrupt":
            m4_interrupt,
    }

    state_tap.STATE_SENDER.send(
        payload
    )

    return result


def main():
    m5.udp_requested_for_time = (
        lockstep_requested_for_time
    )

    state_tap.M7StateUdpSender.ready = (
        never_async_ready
    )

    state_tap.ORIGINAL_STANDALONE_COMPUTE_ACTIONS = (
        lockstep_state_boundary_bridge
    )

    state_tap.m7_udp_env_step = (
        lockstep_m7_udp_env_step
    )

    print("=" * 72)
    print(
        "ICRA27 M7 LOCKSTEP EVALUATION SIDECAR"
    )
    print("=" * 72)
    print(
        "physics steps / HL step : "
        f"{LOCKSTEP_PHYSICS_STEPS}"
    )
    print(
        "wall clock role          : "
        "IPC timeout only"
    )
    print("=" * 72)

    try:
        terrain_seed_runner.main()

    finally:
        m5.udp_requested_for_time = (
            ORIGINAL_UDP_REQUESTED_FOR_TIME
        )

        state_tap.M7StateUdpSender.ready = (
            ORIGINAL_STATE_READY
        )

        state_tap.ORIGINAL_STANDALONE_COMPUTE_ACTIONS = (
            ORIGINAL_STATE_TAP_COMPUTE_TARGET
        )

        state_tap.m7_udp_env_step = (
            ORIGINAL_STATE_TAP_UDP_ENV_STEP
        )


if __name__ == "__main__":
    main()
