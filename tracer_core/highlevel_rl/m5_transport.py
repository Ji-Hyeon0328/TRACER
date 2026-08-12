from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import time
from typing import Any, Sequence

import numpy as np

from tracer_core.highlevel_rl.contracts import (
    normalized_action_to_meta_gait,
    physical_action_to_normalized,
)

from tracer_core.highlevel_rl.state_protocol import (
    decode_state_payload,
)

from tracer_core.lowlevel.pympc_udp_protocol import (
    TELEMETRY_SCHEMA,
    encode_command,
    make_command,
)


COMMAND_KEYS = (
    "vx",
    "yaw_rate",
    "body_height",
    "swing_clearance",
    "gait_period",
    "duty_factor",
)


@dataclass(frozen=True)
class M7MechanicalEnergyInterval:
    """
    Raw physical mechanical-energy increment measured
    during one command interval.

    Reward normalization intentionally does not live
    at the transport layer.
    """

    start_time_s: float
    end_time_s: float
    dt_s: float

    commanded_signed_j: float
    commanded_abs_j: float
    commanded_positive_j: float

    applied_signed_j: float
    applied_abs_j: float
    applied_positive_j: float


@dataclass(frozen=True)
class M7TransportSample:
    command_seq: int

    requested_normalized: tuple[float, ...]
    requested_physical: tuple[float, ...]

    applied_physical: tuple[float, ...] | None

    applied_normalized: tuple[float, ...] | None

    state: dict[str, Any]
    telemetry: dict[str, Any]

    sim_dt_s: float

    energy_interval: (
        M7MechanicalEnergyInterval
        | None
    ) = None

    @property
    def safety_state(self) -> str:
        return str(
            self.telemetry["safety_state"]
        )

    @property
    def override_active(self) -> bool:
        return bool(
            self.telemetry["override_active"]
        )

    @property
    def terminated(self) -> bool:
        return bool(
            self.state["terminated"]
        )

    @property
    def truncated(self) -> bool:
        return bool(
            self.state["truncated"]
        )


def decode_m5_telemetry(
    raw: bytes,
) -> dict[str, Any]:
    payload = json.loads(
        raw.decode("utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "M5 telemetry packet must be object"
        )

    if (
        payload.get("schema")
        != TELEMETRY_SCHEMA
    ):
        raise ValueError(
            "unexpected M5 telemetry schema: "
            f"{payload.get('schema')!r}"
        )

    required = (
        "seq",
        "command_seq",
        "safety_state",
        "override_active",
        "sim_time_s",
        "applied",
    )

    missing = [
        name
        for name in required
        if name not in payload
    ]

    if missing:
        raise ValueError(
            "M5 telemetry missing fields: "
            f"{missing}"
        )

    payload["seq"] = int(
        payload["seq"]
    )

    payload["sim_time_s"] = float(
        payload["sim_time_s"]
    )

    if payload["command_seq"] is not None:
        payload["command_seq"] = int(
            payload["command_seq"]
        )

    safety = str(
        payload["safety_state"]
    ).lower()

    if safety not in {
        "normal",
        "watch",
        "unsafe",
    }:
        raise ValueError(
            f"invalid safety state {safety!r}"
        )

    payload["safety_state"] = safety

    return payload


def _command_tuple(
    mapping: dict[str, Any] | None,
) -> tuple[float, ...] | None:
    if mapping is None:
        return None

    return tuple(
        float(mapping[name])
        for name in COMMAND_KEYS
    )


def _mechanical_energy_interval(
    start_state: dict[str, Any],
    end_state: dict[str, Any],
) -> M7MechanicalEnergyInterval | None:
    """
    Difference two cumulative mechanical-energy snapshots.

    Missing optional energy telemetry preserves backward
    compatibility and returns None.
    """

    start_energy = start_state.get(
        "mechanical_energy"
    )

    end_energy = end_state.get(
        "mechanical_energy"
    )

    start_time = start_state.get(
        "energy_sample_time_s"
    )

    end_time = end_state.get(
        "energy_sample_time_s"
    )

    if (
        start_energy is None
        or end_energy is None
        or start_time is None
        or end_time is None
    ):
        return None

    start_time = float(
        start_time
    )

    end_time = float(
        end_time
    )

    dt = (
        end_time
        - start_time
    )

    if dt <= 0.0:
        return None

    def delta(
        name: str,
    ) -> float:
        return (
            float(end_energy[name])
            - float(start_energy[name])
        )

    commanded_signed = delta(
        "commanded_signed_j"
    )

    commanded_abs = delta(
        "commanded_abs_j"
    )

    commanded_positive = delta(
        "commanded_positive_j"
    )

    applied_signed = delta(
        "applied_signed_j"
    )

    applied_abs = delta(
        "applied_abs_j"
    )

    applied_positive = delta(
        "applied_positive_j"
    )

    tolerance = 1e-9

    for name, value in (
        (
            "commanded_abs_j",
            commanded_abs,
        ),
        (
            "commanded_positive_j",
            commanded_positive,
        ),
        (
            "applied_abs_j",
            applied_abs,
        ),
        (
            "applied_positive_j",
            applied_positive,
        ),
    ):
        if value < -tolerance:
            raise RuntimeError(
                "Cumulative mechanical energy "
                f"decreased for {name}: {value}"
            )

    # Remove possible tiny floating-point negatives.
    commanded_abs = max(
        commanded_abs,
        0.0,
    )

    commanded_positive = max(
        commanded_positive,
        0.0,
    )

    applied_abs = max(
        applied_abs,
        0.0,
    )

    applied_positive = max(
        applied_positive,
        0.0,
    )

    if (
        commanded_positive
        > commanded_abs + tolerance
    ):
        raise RuntimeError(
            "Commanded positive work exceeds "
            "absolute work in interval"
        )

    if (
        applied_positive
        > applied_abs + tolerance
    ):
        raise RuntimeError(
            "Applied positive work exceeds "
            "absolute work in interval"
        )

    return M7MechanicalEnergyInterval(
        start_time_s=start_time,
        end_time_s=end_time,
        dt_s=float(dt),

        commanded_signed_j=float(
            commanded_signed
        ),

        commanded_abs_j=float(
            commanded_abs
        ),

        commanded_positive_j=float(
            commanded_positive
        ),

        applied_signed_j=float(
            applied_signed
        ),

        applied_abs_j=float(
            applied_abs
        ),

        applied_positive_j=float(
            applied_positive
        ),
    )


class M7M5Transport:
    """
    Synchronous M7-v0 transport boundary.

    Policy rate:
        5 Hz nominal

    Command freshness:
        repeat current logical command at 20 Hz

    The same command seq is repeated within one high-level
    action interval so M5 command freshness is maintained
    without turning repeats into new policy decisions.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        command_port: int = 50610,
        telemetry_port: int = 50611,
        state_port: int = 50612,
        command_repeat_hz: float = 20.0,
    ) -> None:
        self.host = str(host)

        self.command_addr = (
            self.host,
            int(command_port),
        )

        self.command_repeat_period_s = (
            1.0
            / max(
                float(command_repeat_hz),
                1.0,
            )
        )

        self.command_sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.telemetry_sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.telemetry_sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        self.telemetry_sock.bind(
            (
                self.host,
                int(telemetry_port),
            )
        )

        self.telemetry_sock.setblocking(
            False
        )

        self.state_sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.state_sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        self.state_sock.bind(
            (
                self.host,
                int(state_port),
            )
        )

        self.state_sock.setblocking(
            False
        )

        self.next_command_seq = 0

        self.latest_telemetry = None
        self.latest_state = None

        self.bad_telemetry_packets = 0
        self.bad_state_packets = 0

    def close(self) -> None:
        self.command_sock.close()
        self.telemetry_sock.close()
        self.state_sock.close()

    def _drain_telemetry(
        self,
    ) -> list[dict[str, Any]]:
        packets = []

        while True:
            try:
                raw, _ = (
                    self.telemetry_sock.recvfrom(
                        65535
                    )
                )

            except BlockingIOError:
                break

            try:
                payload = (
                    decode_m5_telemetry(raw)
                )

            except Exception:
                self.bad_telemetry_packets += 1
                continue

            self.latest_telemetry = payload
            packets.append(payload)

        return packets

    def _drain_state(
        self,
    ) -> list[dict[str, Any]]:
        packets = []

        while True:
            try:
                raw, _ = (
                    self.state_sock.recvfrom(
                        65535
                    )
                )

            except BlockingIOError:
                break

            try:
                payload = (
                    decode_state_payload(raw)
                )

            except Exception:
                self.bad_state_packets += 1
                continue

            self.latest_state = payload
            packets.append(payload)

        return packets

    def poll(self) -> None:
        self._drain_telemetry()
        self._drain_state()

    def wait_initial(
        self,
        *,
        timeout_s: float = 15.0,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
    ]:
        deadline = (
            time.monotonic()
            + float(timeout_s)
        )

        while time.monotonic() < deadline:
            self.poll()

            if (
                self.latest_telemetry
                is not None
                and self.latest_state
                is not None
            ):
                return (
                    self.latest_telemetry,
                    self.latest_state,
                )

            time.sleep(0.005)

        raise TimeoutError(
            "Timed out waiting for initial "
            "M5 telemetry + M7 state"
        )

    def step(
        self,
        normalized_action: Sequence[float],
        *,
        target_sim_dt: float = 0.20,
        timeout_s: float = 10.0,
    ) -> M7TransportSample:
        action = np.asarray(
            normalized_action,
            dtype=np.float64,
        ).reshape(-1)

        if action.shape != (4,):
            raise ValueError(
                "normalized_action must "
                f"have shape (4,), got {action.shape}"
            )

        policy_command = (
            normalized_action_to_meta_gait(
                action
            )
        )

        seq = self.next_command_seq
        self.next_command_seq += 1

        udp_command = make_command(
            seq=seq,
            label=f"m7_policy_{seq}",
            vx=policy_command.vx,
            yaw_rate=policy_command.yaw_rate,
            body_height=(
                policy_command.body_height
            ),
            swing_clearance=(
                policy_command.swing_clearance
            ),
            gait_period=(
                policy_command.gait_period
            ),
            duty_factor=(
                policy_command.duty_factor
            ),
        )

        raw_command = encode_command(
            udp_command
        )

        requested_physical = tuple(
            float(x)
            for x in udp_command.values
        )

        deadline = (
            time.monotonic()
            + float(timeout_s)
        )

        next_send = 0.0

        ack_sim_time = None
        target_sim_time = None

        matched_telemetry = None
        matched_state = None

        energy_start_state = None

        # ----------------------------------------------------
        # One logical M7 command must never span an internal
        # simulator reset.
        #
        # The state protocol already carries episode_index.
        # Anchor this command to the episode that was active
        # when step() began.
        # ----------------------------------------------------
        if self.latest_state is None:
            raise RuntimeError(
                "M7 transport step has no initial state"
            )

        step_entry_state = self.latest_state

        active_episode_index = int(
            self.latest_state["episode_index"]
        )

        last_active_state = self.latest_state
        latest_seq_telemetry = None
        terminal_state = None
        reset_boundary_seen = False

        while time.monotonic() < deadline:
            now = time.monotonic()

            if now >= next_send:
                self.command_sock.sendto(
                    raw_command,
                    self.command_addr,
                )

                next_send = (
                    now
                    + self.command_repeat_period_s
                )

            telemetry_packets = (
                self._drain_telemetry()
            )

            state_packets = (
                self._drain_state()
            )

            active_state_packets = []

            for state in state_packets:
                state_episode_index = int(
                    state["episode_index"]
                )

                if (
                    state_episode_index
                    != active_episode_index
                ):
                    reset_boundary_seen = True
                    continue

                active_state_packets.append(
                    state
                )

                last_active_state = state

                if (
                    bool(state["terminated"])
                    or bool(state["truncated"])
                ):
                    terminal_state = state

            for telemetry in telemetry_packets:
                if (
                    telemetry["command_seq"]
                    != seq
                ):
                    continue

                latest_seq_telemetry = telemetry

                if (
                    str(
                        telemetry["safety_state"]
                    ).strip().lower()
                    == "unsafe"
                    or bool(
                        telemetry["override_active"]
                    )
                ):
                    # M4 is a hard execution boundary.
                    #
                    # Do not wait for the nominal target time
                    # after safety termination.
                    matched_telemetry = telemetry

                if ack_sim_time is None:
                    ack_sim_time = float(
                        telemetry["sim_time_s"]
                    )

                    target_sim_time = (
                        ack_sim_time
                        + float(target_sim_dt)
                    )

                if (
                    target_sim_time
                    is not None
                    and float(
                        telemetry["sim_time_s"]
                    )
                    >= (
                        target_sim_time - 0.06
                    )
                ):
                    matched_telemetry = telemetry

            if (
                ack_sim_time is not None
                and energy_start_state is None
            ):
                for state in active_state_packets:
                    energy = state.get(
                        "mechanical_energy"
                    )

                    energy_time = state.get(
                        "energy_sample_time_s"
                    )

                    if (
                        energy is None
                        or energy_time is None
                    ):
                        continue

                    if (
                        float(energy_time)
                        >= float(ack_sim_time)
                    ):
                        energy_start_state = (
                            state
                        )

                        break

            # ------------------------------------------------
            # Terminal events end the logical command interval
            # immediately.  A 5-Hz target duration is not a
            # requirement after native/M4 termination.
            # ------------------------------------------------
            terminal_telemetry = (
                latest_seq_telemetry is not None
                and (
                    str(
                        latest_seq_telemetry[
                            "safety_state"
                        ]
                    ).strip().lower()
                    == "unsafe"
                    or bool(
                        latest_seq_telemetry[
                            "override_active"
                        ]
                    )
                )
            )

            if terminal_state is not None:
                matched_state = terminal_state

                if latest_seq_telemetry is not None:
                    matched_telemetry = (
                        latest_seq_telemetry
                    )

            elif (
                terminal_telemetry
                and ack_sim_time is not None
                and last_active_state is not None
                and float(
                    last_active_state[
                        "sample_time_s"
                    ]
                )
                >= float(ack_sim_time)
            ):
                # A safety telemetry packet can arrive before
                # the terminal state packet is emitted at the
                # state publication rate.  Use the newest
                # physical state from the same episode.
                matched_state = last_active_state
                matched_telemetry = (
                    latest_seq_telemetry
                )

            elif target_sim_time is not None:
                for state in active_state_packets:
                    if float(
                        state["sample_time_s"]
                    ) >= target_sim_time:
                        matched_state = state

            # Also allow an already-buffered latest sample,
            # but only from this command's original episode.
            if (
                matched_state is None
                and target_sim_time is not None
                and self.latest_state is not None
                and int(
                    self.latest_state[
                        "episode_index"
                    ]
                )
                == active_episode_index
                and float(
                    self.latest_state[
                        "sample_time_s"
                    ]
                )
                >= target_sim_time
            ):
                matched_state = (
                    self.latest_state
                )

            if (
                matched_telemetry is None
                and target_sim_time is not None
                and self.latest_telemetry
                is not None
                and self.latest_telemetry[
                    "command_seq"
                ] == seq
                and float(
                    self.latest_telemetry[
                        "sim_time_s"
                    ]
                )
                >= (
                    target_sim_time - 0.06
                )
            ):
                matched_telemetry = (
                    self.latest_telemetry
                )

            if (
                matched_state is not None
                and matched_telemetry is not None
            ):
                break

            if reset_boundary_seen:
                raise RuntimeError(
                    "M7 simulator episode changed during "
                    f"command seq={seq}: "
                    f"active_episode="
                    f"{active_episode_index}. "
                    "Refusing cross-episode state/energy "
                    "matching."
                )

            time.sleep(0.002)

        if ack_sim_time is None:
            raise TimeoutError(
                f"M5 never acknowledged "
                f"command seq={seq}"
            )

        if matched_state is None:
            raise TimeoutError(
                "No physical-state sample reached "
                f"target sim time for seq={seq}"
            )

        if matched_telemetry is None:
            raise TimeoutError(
                "No matching M5 telemetry reached "
                f"target sim time for seq={seq}"
            )

        applied_physical = _command_tuple(
            matched_telemetry.get(
                "applied"
            )
        )

        applied_normalized = None

        if applied_physical is not None:
            applied_normalized = tuple(
                float(x)
                for x in (
                    physical_action_to_normalized(
                        applied_physical[:4]
                    )
                )
            )

        terminal_transition = bool(
            matched_state["terminated"]
            or matched_state["truncated"]
            or (
                str(
                    matched_telemetry[
                        "safety_state"
                    ]
                ).strip().lower()
                == "unsafe"
            )
            or bool(
                matched_telemetry[
                    "override_active"
                ]
            )
        )

        # ----------------------------------------------------
        # Decision-time semantics.
        #
        # State sample_time_s is captured before the MuJoCo
        # env step. A terminal event may therefore occur in
        # that step while the state timestamp still equals
        # the command ACK timestamp.
        #
        # Normal transitions retain the existing state-time
        # semantics. For terminal transitions only, recover
        # a positive end boundary from post-step energy and/or
        # matching M5 safety telemetry when necessary.
        # ----------------------------------------------------
        state_end_time = float(
            matched_state[
                "sample_time_s"
            ]
        )

        sim_dt = (
            state_end_time
            - float(ack_sim_time)
        )

        if (
            terminal_transition
            and sim_dt <= 0.0
        ):
            terminal_end_candidates = []

            energy_end_time = (
                matched_state.get(
                    "energy_sample_time_s"
                )
            )

            if (
                energy_end_time is not None
                and float(energy_end_time)
                > float(ack_sim_time)
            ):
                terminal_end_candidates.append(
                    float(energy_end_time)
                )

            telemetry_end_time = (
                matched_telemetry.get(
                    "sim_time_s"
                )
            )

            if (
                telemetry_end_time is not None
                and float(telemetry_end_time)
                > float(ack_sim_time)
            ):
                terminal_end_candidates.append(
                    float(telemetry_end_time)
                )

            if not terminal_end_candidates:
                raise RuntimeError(
                    "Terminal M7 transition has no "
                    "positive-duration boundary: "
                    f"ack={ack_sim_time} "
                    f"state={state_end_time}"
                )

            terminal_end_time = min(
                terminal_end_candidates
            )

            sim_dt = (
                terminal_end_time
                - float(ack_sim_time)
            )

        if sim_dt <= 0.0:
            raise RuntimeError(
                "M7 transport produced non-positive "
                "decision duration: "
                f"seq={seq} "
                f"ack={ack_sim_time} "
                f"state_end={state_end_time} "
                f"dt={sim_dt}"
            )

        energy_interval = None

        selected_energy_start_state = (
            energy_start_state
        )

        # ----------------------------------------------------
        # Sparse terminal-state publication edge case:
        #
        # The first state carrying energy after command ACK can
        # itself be the terminal state. In that case the normal
        # post-ACK selector chooses the same snapshot as both
        # interval start and end, yielding dt_E == 0.
        #
        # For terminal transitions only, fall back to the last
        # pre-command energy snapshot from the same simulator
        # episode. This forms a positive-duration bracket around
        # the terminal event without weakening energy
        # monotonicity or permitting cross-episode differencing.
        # ----------------------------------------------------
        if terminal_transition:
            end_energy_time = (
                matched_state.get(
                    "energy_sample_time_s"
                )
            )

            selected_start_time = None

            if (
                selected_energy_start_state
                is not None
            ):
                selected_start_time = (
                    selected_energy_start_state.get(
                        "energy_sample_time_s"
                    )
                )

            needs_terminal_bracket = bool(
                end_energy_time is not None
                and (
                    selected_start_time is None
                    or float(
                        selected_start_time
                    )
                    >= float(
                        end_energy_time
                    ) - 1e-12
                )
            )

            if needs_terminal_bracket:
                candidate = step_entry_state

                candidate_energy = (
                    None
                    if candidate is None
                    else candidate.get(
                        "mechanical_energy"
                    )
                )

                candidate_time = (
                    None
                    if candidate is None
                    else candidate.get(
                        "energy_sample_time_s"
                    )
                )

                if (
                    candidate is not None
                    and int(
                        candidate[
                            "episode_index"
                        ]
                    )
                    == active_episode_index
                    and candidate_energy
                    is not None
                    and candidate_time
                    is not None
                    and float(candidate_time)
                    < float(end_energy_time)
                    - 1e-12
                ):
                    selected_energy_start_state = (
                        candidate
                    )

        if (
            selected_energy_start_state
            is not None
        ):
            start_episode_index = int(
                selected_energy_start_state[
                    "episode_index"
                ]
            )

            end_episode_index = int(
                matched_state[
                    "episode_index"
                ]
            )

            if (
                start_episode_index
                != end_episode_index
            ):
                raise RuntimeError(
                    "Mechanical-energy interval crossed "
                    "simulator episode boundary: "
                    f"start={start_episode_index} "
                    f"end={end_episode_index}"
                )

            energy_interval = (
                _mechanical_energy_interval(
                    selected_energy_start_state,
                    matched_state,
                )
            )

        if (
            terminal_transition
            and energy_interval is None
        ):
            raise RuntimeError(
                "Terminal M7 transition has no "
                "positive-duration same-episode "
                "mechanical-energy bracket"
            )

        return M7TransportSample(
            command_seq=seq,

            requested_normalized=tuple(
                float(x)
                for x in action
            ),

            requested_physical=(
                requested_physical
            ),

            applied_physical=(
                applied_physical
            ),

            applied_normalized=(
                applied_normalized
            ),

            state=matched_state,
            telemetry=matched_telemetry,

            sim_dt_s=float(sim_dt),

            energy_interval=(
                energy_interval
            ),
        )
