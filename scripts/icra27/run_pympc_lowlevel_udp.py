#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
ICRA27 = ROOT / "scripts" / "icra27"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ICRA27))


import run_pympc_lowlevel_standalone as standalone

from tracer_core.highlevel.meta_gait import (
    MetaGaitCommand,
)

from tracer_core.lowlevel.pympc_lowlevel_runtime import (
    PyMPCLowLevelRuntime,
)

from tracer_core.lowlevel.pympc_udp_protocol import (
    decode_command,
    encode_telemetry,
    make_telemetry_payload,
)


NOMINAL = standalone.NOMINAL


class MetaGaitUdpInbox:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_s: float,
    ):
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(
            timeout_s
        )

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        self.sock.bind(
            (
                self.host,
                self.port,
            )
        )

        self.sock.setblocking(False)

        self.latest = None

        self.latest_receive_time = None

        self.last_announced_seq = None

        self.bad_packet_count = 0

    def poll(self):
        while True:
            try:
                raw, _ = (
                    self.sock.recvfrom(
                        65535
                    )
                )

            except BlockingIOError:
                break

            try:
                command = decode_command(
                    raw
                )

            except Exception as exc:
                self.bad_packet_count += 1

                if (
                    self.bad_packet_count
                    <= 3
                ):
                    print(
                        "[M5 UDP] bad command "
                        f"packet: {exc}"
                    )

                continue

            if (
                self.latest is not None
                and command.seq
                < self.latest.seq
            ):
                continue

            self.latest = command

            self.latest_receive_time = (
                time.monotonic()
            )

            if (
                command.seq
                != self.last_announced_seq
            ):
                print(
                    "[M5 UDP] accepted "
                    f"seq={command.seq} "
                    f"vx={command.vx:.3f} "
                    f"yaw="
                    f"{command.yaw_rate:.3f} "
                    f"h="
                    f"{command.body_height:.3f} "
                    f"clr="
                    f"{command.swing_clearance:.3f} "
                    f"T="
                    f"{command.gait_period:.3f} "
                    f"D="
                    f"{command.duty_factor:.3f}"
                )

                self.last_announced_seq = (
                    command.seq
                )

    def age_s(self):
        if (
            self.latest_receive_time
            is None
        ):
            return None

        return max(
            0.0,
            time.monotonic()
            - self.latest_receive_time,
        )

    def is_fresh(self) -> bool:
        age = self.age_s()

        return (
            age is not None
            and age <= self.timeout_s
        )

    def current_request(
        self,
    ) -> tuple[
        str,
        MetaGaitCommand,
    ]:
        self.poll()

        if (
            self.latest is None
            or not self.is_fresh()
        ):
            return (
                "transport_nominal",
                NOMINAL,
            )

        p = self.latest

        command = MetaGaitCommand(
            vx=p.vx,
            yaw_rate=p.yaw_rate,
            body_height=p.body_height,
            swing_clearance=(
                p.swing_clearance
            ),
            gait_period=p.gait_period,
            duty_factor=p.duty_factor,
            source_mode="ros2_udp",
            source_reason=(
                f"UDP command seq={p.seq}"
            ),
        )

        return (
            f"ros2_seq_{p.seq}",
            command,
        )

    def close(self):
        self.sock.close()


class TelemetryUdpSender:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        telemetry_hz: float,
    ):
        self.addr = (
            str(host),
            int(port),
        )

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.period_s = (
            1.0
            / max(
                float(telemetry_hz),
                1.0,
            )
        )

        self.last_send = 0.0
        self.seq = 0

    def ready(self) -> bool:
        return (
            time.monotonic()
            - self.last_send
            >= self.period_s
        )

    def send(
        self,
        payload,
    ):
        self.sock.sendto(
            encode_telemetry(payload),
            self.addr,
        )

        self.last_send = time.monotonic()
        self.seq += 1

    def close(self):
        self.sock.close()


INBOX: MetaGaitUdpInbox | None = None
TELEMETRY: TelemetryUdpSender | None = None

CURRENT_REQUESTED_LABEL = (
    "transport_nominal"
)

CURRENT_SIM_TIME_S = 0.0

ORIGINAL_REQUESTED_FOR_TIME = (
    standalone.requested_for_time
)

ORIGINAL_STANDALONE_ENV_STEP = (
    standalone.standalone_env_step
)


def udp_requested_for_time(
    time_s: float,
):
    global CURRENT_REQUESTED_LABEL
    global CURRENT_SIM_TIME_S

    CURRENT_SIM_TIME_S = float(time_s)

    assert INBOX is not None

    (
        label,
        command,
    ) = INBOX.current_request()

    CURRENT_REQUESTED_LABEL = label

    return label, command


def udp_env_step(
    self,
    action,
):
    result = ORIGINAL_STANDALONE_ENV_STEP(
        self,
        action,
    )

    assert INBOX is not None
    assert TELEMETRY is not None
    assert standalone.RUNTIME is not None

    if not TELEMETRY.ready():
        return result

    runtime = standalone.RUNTIME

    latest = INBOX.latest

    payload = make_telemetry_payload(
        seq=TELEMETRY.seq,
        command_seq=(
            None
            if latest is None
            else latest.seq
        ),
        requested_label=(
            CURRENT_REQUESTED_LABEL
        ),
        selected_label=(
            runtime.target_label
        ),
        safety_state=(
            runtime
            .safety_monitor
            .last_status
            .state
            .value
        ),
        override_active=(
            runtime
            .supervisor
            .override_active
        ),
        override_reasons=(
            runtime
            .supervisor
            .override_reasons
        ),
        structural_commit=(
            runtime
            .transition_manager
            .last_structural_commit
        ),
        sim_time_s=CURRENT_SIM_TIME_S,
        structural_commit_count=(
            runtime
            .transition_manager
            .structural_commit_count
        ),
        command_fresh=(
            INBOX.is_fresh()
        ),
        command_age_s=(
            INBOX.age_s()
        ),
        applied=runtime.command,
    )

    TELEMETRY.send(payload)

    return result


def clear_standalone_state():
    global CURRENT_REQUESTED_LABEL
    global CURRENT_SIM_TIME_S

    CURRENT_REQUESTED_LABEL = (
        "transport_nominal"
    )
    CURRENT_SIM_TIME_S = 0.0

    standalone.PENDING_SAMPLE = None
    standalone.LAST_REQUESTED_LABEL = None
    standalone.LAST_SAFETY_STATE = None
    standalone.LAST_COMMAND_SIGNATURE = None

    standalone.FIRST_UNSAFE = None
    standalone.FIRST_OVERRIDE = None
    standalone.FIRST_BACKOFF_COMMIT = None

    standalone.STRUCTURAL_EVENTS.clear()
    standalone.STATE_EVENTS.clear()
    standalone.COMMAND_EVENTS.clear()
    standalone.TERMINATION_EVENTS.clear()
    standalone.RESET_EVENTS.clear()


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--command-host",
        default="127.0.0.1",
    )

    p.add_argument(
        "--command-port",
        type=int,
        default=50510,
    )

    p.add_argument(
        "--telemetry-host",
        default="127.0.0.1",
    )

    p.add_argument(
        "--telemetry-port",
        type=int,
        default=50511,
    )

    p.add_argument(
        "--command-timeout",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--telemetry-hz",
        type=float,
        default=50.0,
    )

    p.add_argument(
        "--duration",
        type=float,
        default=12.0,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    p.add_argument(
        "--no-render",
        action="store_true",
    )

    p.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = p.parse_args()

    if args.duration <= 0.0:
        p.error(
            "--duration must be > 0"
        )

    if args.command_timeout <= 0.0:
        p.error(
            "--command-timeout must be > 0"
        )

    if args.telemetry_hz <= 0.0:
        p.error(
            "--telemetry-hz must be > 0"
        )

    return args


def main():
    global INBOX
    global TELEMETRY

    args = parse_args()

    INBOX = MetaGaitUdpInbox(
        host=args.command_host,
        port=args.command_port,
        timeout_s=args.command_timeout,
    )

    TELEMETRY = TelemetryUdpSender(
        host=args.telemetry_host,
        port=args.telemetry_port,
        telemetry_hz=args.telemetry_hz,
    )

    clear_standalone_state()

    standalone.RUNTIME = (
        PyMPCLowLevelRuntime(
            fallback_command=NOMINAL,
        )
    )

    # These are kept populated because the
    # canonical standalone hook owns them,
    # although UDP command selection replaces
    # the original time schedule.
    standalone.TARGET = NOMINAL

    standalone.ARGS = SimpleNamespace(
        nominal_duration=0.0,
    )

    standalone.requested_for_time = (
        udp_requested_for_time
    )

    standalone.cfg.simulation_params[
        "gait"
    ] = "trot"

    standalone.cfg.mpc_params[
        "optimize_step_freq"
    ] = False

    print("=" * 72)
    print(
        "ICRA27 M5 ROS2/UDP -> "
        "CLOSED PyMPC LOW-LEVEL"
    )
    print("=" * 72)

    print(
        "command UDP   : "
        f"{args.command_host}:"
        f"{args.command_port}"
    )

    print(
        "telemetry UDP : "
        f"{args.telemetry_host}:"
        f"{args.telemetry_port}"
    )

    print(
        "command timeout: "
        f"{args.command_timeout:.3f}s"
    )

    print(
        "transport fallback: "
        "known nominal command"
    )

    print(
        "NOTE: transport fallback "
        "still passes through "
        "PyMPCLowLevelRuntime."
    )

    print("=" * 72)

    (
        standalone.sim
        .QuadrupedPyMPC_Wrapper
        .compute_actions
    ) = standalone.standalone_compute_actions

    standalone.QuadrupedEnv.step = (
        udp_env_step
    )

    (
        standalone.sim
        .QuadrupedPyMPC_Wrapper
        .reset
    ) = standalone.standalone_wrapper_reset

    hooks_restored = False

    try:
        standalone.sim.run_simulation(
            qpympc_cfg=standalone.cfg,
            num_episodes=1,
            num_seconds_per_episode=(
                args.duration
            ),
            ref_base_lin_vel=(
                NOMINAL.vx
                / float(
                    standalone.cfg.hip_height
                )
            ),
            ref_base_ang_vel=0.0,
            friction_coeff=0.8,
            base_vel_command_type="forward",
            seed=args.seed,
            render=not args.no_render,
            recording_path=None,
        )

    finally:
        (
            standalone.sim
            .QuadrupedPyMPC_Wrapper
            .compute_actions
        ) = standalone.ORIGINAL_COMPUTE_ACTIONS

        standalone.QuadrupedEnv.step = (
            standalone.ORIGINAL_ENV_STEP
        )

        (
            standalone.sim
            .QuadrupedPyMPC_Wrapper
            .reset
        ) = standalone.ORIGINAL_WRAPPER_RESET

        standalone.requested_for_time = (
            ORIGINAL_REQUESTED_FOR_TIME
        )

        hooks_restored = (
            standalone.sim
            .QuadrupedPyMPC_Wrapper
            .compute_actions
            is standalone.ORIGINAL_COMPUTE_ACTIONS

            and standalone.QuadrupedEnv.step
            is standalone.ORIGINAL_ENV_STEP

            and standalone.sim
            .QuadrupedPyMPC_Wrapper
            .reset
            is standalone.ORIGINAL_WRAPPER_RESET

            and standalone.requested_for_time
            is ORIGINAL_REQUESTED_FOR_TIME
        )

        if INBOX is not None:
            INBOX.close()

        if TELEMETRY is not None:
            TELEMETRY.close()

    result = {
        "mode":
            "pympc_lowlevel_udp_v0",

        "command_host":
            args.command_host,

        "command_port":
            args.command_port,

        "telemetry_host":
            args.telemetry_host,

        "telemetry_port":
            args.telemetry_port,

        "command_timeout_s":
            args.command_timeout,

        "last_received_command_seq": (
            None
            if INBOX is None
            or INBOX.latest is None
            else INBOX.latest.seq
        ),

        "bad_packet_count": (
            None
            if INBOX is None
            else INBOX.bad_packet_count
        ),

        "first_unsafe":
            standalone.FIRST_UNSAFE,

        "first_override":
            standalone.FIRST_OVERRIDE,

        "first_backoff_commit":
            standalone.FIRST_BACKOFF_COMMIT,

        "termination_count":
            len(
                standalone
                .TERMINATION_EVENTS
            ),

        "reset_count":
            len(
                standalone.RESET_EVENTS
            ),

        "structural_events":
            standalone.STRUCTURAL_EVENTS,

        "command_events":
            standalone.COMMAND_EVENTS,

        "hooks_restored":
            hooks_restored,
    }

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(
                result,
                indent=2,
            )
            + "\n"
        )

    print()
    print("=" * 72)
    print("M5 UDP RUNTIME RESULT")
    print("=" * 72)

    print(
        "last command seq :",
        result[
            "last_received_command_seq"
        ],
    )

    print(
        "bad packets      :",
        result["bad_packet_count"],
    )

    print(
        "UNSAFE           :",
        (
            None
            if standalone.FIRST_UNSAFE
            is None
            else standalone.FIRST_UNSAFE[
                "time_s"
            ]
        ),
    )

    print(
        "override         :",
        standalone.FIRST_OVERRIDE
        is not None,
    )

    print(
        "terminations     :",
        result["termination_count"],
    )

    print(
        "hooks restored   :",
        hooks_restored,
    )

    if args.output is not None:
        print(
            "saved            :",
            args.output,
        )

    print("=" * 72)

    if not hooks_restored:
        raise RuntimeError(
            "M5 UDP hooks were not restored"
        )


if __name__ == "__main__":
    main()
