#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tracer_core.highlevel_rl.m5_transport import (
    M7M5Transport,
    _mechanical_energy_interval,
)


class FakeCommandSocket:
    def sendto(self, raw, addr):
        del raw, addr

    def close(self):
        pass


def energy(
    *,
    t,
    commanded_abs,
    commanded_positive,
    applied_abs,
    applied_positive,
):
    return {
        "samples": int(round(t * 500.0)),
        "elapsed_s": float(t),

        "commanded_signed_j":
            float(commanded_positive),
        "commanded_abs_j":
            float(commanded_abs),
        "commanded_positive_j":
            float(commanded_positive),

        "applied_signed_j":
            float(applied_positive),
        "applied_abs_j":
            float(applied_abs),
        "applied_positive_j":
            float(applied_positive),
    }


def state(
    *,
    episode,
    t,
    commanded_abs,
    commanded_positive,
    applied_abs,
    applied_positive,
    terminated=False,
    truncated=False,
):
    return {
        "episode_index": int(episode),

        "sample_time_s": float(t),
        "energy_sample_time_s": float(t),

        "mechanical_energy": energy(
            t=t,
            commanded_abs=commanded_abs,
            commanded_positive=(
                commanded_positive
            ),
            applied_abs=applied_abs,
            applied_positive=(
                applied_positive
            ),
        ),

        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def telemetry(
    *,
    seq,
    t,
    safety="normal",
    override=False,
):
    return {
        "command_seq": int(seq),
        "sim_time_s": float(t),

        "safety_state": str(safety),
        "override_active": bool(override),

        "applied": None,
    }


def make_transport():
    transport = M7M5Transport(
        host="127.0.0.1",

        command_port=9,

        telemetry_port=0,
        state_port=0,

        command_repeat_hz=20.0,
    )

    transport.command_sock.close()
    transport.command_sock = (
        FakeCommandSocket()
    )

    return transport


def check_terminal_partial_interval():
    transport = make_transport()

    try:
        start = state(
            episode=-1,
            t=3.660,

            commanded_abs=100.0,
            commanded_positive=60.0,

            applied_abs=200.0,
            applied_positive=120.0,
        )

        terminal = state(
            episode=-1,
            t=3.686,

            commanded_abs=101.0,
            commanded_positive=60.7,

            applied_abs=201.2,
            applied_positive=120.8,

            terminated=True,
        )

        transport.latest_state = start

        telemetry_batches = [
            [
                telemetry(
                    seq=0,
                    t=3.660,
                )
            ],
            [
                telemetry(
                    seq=0,
                    t=3.686,
                    safety="unsafe",
                    override=True,
                )
            ],
        ]

        state_batches = [
            [start],
            [terminal],
        ]

        def drain_telemetry():
            if telemetry_batches:
                batch = telemetry_batches.pop(0)
            else:
                batch = []

            if batch:
                transport.latest_telemetry = (
                    batch[-1]
                )

            return batch

        def drain_state():
            if state_batches:
                batch = state_batches.pop(0)
            else:
                batch = []

            if batch:
                transport.latest_state = batch[-1]

            return batch

        transport._drain_telemetry = (
            drain_telemetry
        )

        transport._drain_state = (
            drain_state
        )

        sample = transport.step(
            np.zeros(
                4,
                dtype=np.float64,
            ),
            target_sim_dt=0.20,
            timeout_s=1.0,
        )

        if sample.state["episode_index"] != -1:
            raise RuntimeError(
                "Terminal state episode mismatch"
            )

        if not sample.state["terminated"]:
            raise RuntimeError(
                "Terminal state was not returned"
            )

        if (
            sample.safety_state.strip().lower()
            != "unsafe"
        ):
            raise RuntimeError(
                "Unsafe telemetry was not returned"
            )

        if sample.energy_interval is None:
            raise RuntimeError(
                "Terminal transition lost "
                "energy interval"
            )

        if not math.isclose(
            sample.sim_dt_s,
            0.026,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "Expected partial terminal "
                f"dt=0.026, got "
                f"{sample.sim_dt_s}"
            )

        if not math.isclose(
            sample.energy_interval
            .applied_abs_j,
            1.2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "Unexpected partial applied work: "
                f"{sample.energy_interval.applied_abs_j}"
            )

        print(
            "terminal partial interval : PASS "
            f"dt={sample.sim_dt_s:.6f}s "
            f"Eabs="
            f"{sample.energy_interval.applied_abs_j:.6f}J"
        )

    finally:
        transport.close()


def check_reset_boundary_rejected():
    transport = make_transport()

    try:
        start = state(
            episode=-1,
            t=3.660,

            commanded_abs=100.0,
            commanded_positive=60.0,

            applied_abs=200.0,
            applied_positive=120.0,
        )

        post_reset = state(
            episode=0,
            t=0.010,

            commanded_abs=0.5,
            commanded_positive=0.3,

            applied_abs=0.8,
            applied_positive=0.5,
        )

        transport.latest_state = start

        telemetry_batches = [
            [
                telemetry(
                    seq=0,
                    t=3.660,
                )
            ],
            [
                telemetry(
                    seq=0,
                    t=3.700,
                )
            ],
        ]

        state_batches = [
            [start],
            [post_reset],
        ]

        def drain_telemetry():
            if telemetry_batches:
                batch = telemetry_batches.pop(0)
            else:
                batch = []

            if batch:
                transport.latest_telemetry = (
                    batch[-1]
                )

            return batch

        def drain_state():
            if state_batches:
                batch = state_batches.pop(0)
            else:
                batch = []

            if batch:
                transport.latest_state = batch[-1]

            return batch

        transport._drain_telemetry = (
            drain_telemetry
        )

        transport._drain_state = (
            drain_state
        )

        try:
            transport.step(
                np.zeros(
                    4,
                    dtype=np.float64,
                ),
                target_sim_dt=0.20,
                timeout_s=1.0,
            )

        except RuntimeError as exc:
            message = str(exc)

            if (
                "simulator episode changed"
                not in message
            ):
                raise

            print(
                "cross-episode rejection    : PASS"
            )

        else:
            raise RuntimeError(
                "Cross-episode state was not "
                "rejected"
            )

    finally:
        transport.close()


def check_monotonicity_guard_preserved():
    start = state(
        episode=0,
        t=1.000,

        commanded_abs=10.0,
        commanded_positive=6.0,

        applied_abs=20.0,
        applied_positive=12.0,
    )

    end = state(
        episode=0,
        t=1.200,

        commanded_abs=9.0,
        commanded_positive=5.5,

        applied_abs=19.0,
        applied_positive=11.5,
    )

    try:
        _mechanical_energy_interval(
            start,
            end,
        )

    except RuntimeError as exc:
        if (
            "Cumulative mechanical energy "
            "decreased"
            not in str(exc)
        ):
            raise

        print(
            "strict energy monotonicity  : PASS"
        )

    else:
        raise RuntimeError(
            "Negative cumulative-energy "
            "delta was incorrectly accepted"
        )


def check_terminal_single_post_ack_energy_state():
    """
    Reproduce the sparse terminal-publication edge case:

      command ACK
          |
          |  no intermediate energy state
          v
      first post-ACK energy state == terminal state

    The ordinary post-ACK selector therefore initially chooses
    the terminal state as both energy start and end.

    Transport must fall back to the last pre-command
    same-episode energy snapshot.
    """
    transport = make_transport()

    try:
        pre_command = state(
            episode=-1,
            t=3.650,

            commanded_abs=100.0,
            commanded_positive=60.0,

            applied_abs=200.0,
            applied_positive=120.0,
        )

        terminal = state(
            episode=-1,
            t=3.686,

            commanded_abs=101.0,
            commanded_positive=60.7,

            applied_abs=201.2,
            applied_positive=120.8,

            terminated=True,
        )

        transport.latest_state = pre_command

        telemetry_batches = [
            [
                telemetry(
                    seq=0,
                    t=3.660,
                )
            ],
            [
                telemetry(
                    seq=0,
                    t=3.686,
                    safety="unsafe",
                    override=True,
                )
            ],
        ]

        # Critically, there is NO ordinary post-ACK
        # energy state before the terminal state.
        state_batches = [
            [],
            [terminal],
        ]

        def drain_telemetry():
            if telemetry_batches:
                batch = telemetry_batches.pop(0)
            else:
                batch = []

            if batch:
                transport.latest_telemetry = (
                    batch[-1]
                )

            return batch

        def drain_state():
            if state_batches:
                batch = state_batches.pop(0)
            else:
                batch = []

            if batch:
                transport.latest_state = batch[-1]

            return batch

        transport._drain_telemetry = (
            drain_telemetry
        )

        transport._drain_state = (
            drain_state
        )

        sample = transport.step(
            np.zeros(
                4,
                dtype=np.float64,
            ),
            target_sim_dt=0.20,
            timeout_s=1.0,
        )

        if sample.energy_interval is None:
            raise RuntimeError(
                "Sparse terminal transition "
                "lost energy interval"
            )

        if not math.isclose(
            sample.energy_interval.dt_s,
            0.036,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "Unexpected terminal bracket dt: "
                f"{sample.energy_interval.dt_s}"
            )

        if not math.isclose(
            sample.energy_interval.applied_abs_j,
            1.2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "Unexpected terminal bracket work: "
                f"{sample.energy_interval.applied_abs_j}"
            )

        print(
            "sparse terminal bracket    : PASS "
            f"dtE={sample.energy_interval.dt_s:.6f}s "
            f"Eabs="
            f"{sample.energy_interval.applied_abs_j:.6f}J"
        )

    finally:
        transport.close()


def main():
    print(
        "# ICRA27 M7 TRANSPORT "
        "TERMINAL-BOUNDARY CHECK"
    )
    print()

    check_terminal_partial_interval()
    check_terminal_single_post_ack_energy_state()
    check_reset_boundary_rejected()
    check_monotonicity_guard_preserved()

    print()
    print(
        "[ICRA27] M7 transport terminal/reset "
        "boundary: PASS"
    )


if __name__ == "__main__":
    main()
