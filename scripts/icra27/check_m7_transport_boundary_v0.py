#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.highlevel_rl.m5_transport import (
    M7M5Transport,
)


ACTIONS = [
    (
        "nominal",
        [0.0, 0.0, 0.0, 0.0],
    ),
    (
        "faster",
        [0.20, 0.0, 0.0, 0.0],
    ),
    (
        "slow_turn",
        [-0.20, 0.10, 0.0, 0.0],
    ),
    (
        "geometry",
        [0.0, 0.0, 0.10, -0.10],
    ),
]


def fmt(values):
    if values is None:
        return "None"

    return "[" + ", ".join(
        f"{float(x):+.4f}"
        for x in values
    ) + "]"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--host",
        default="127.0.0.1",
    )

    ap.add_argument(
        "--command-port",
        type=int,
        default=50610,
    )

    ap.add_argument(
        "--telemetry-port",
        type=int,
        default=50611,
    )

    ap.add_argument(
        "--state-port",
        type=int,
        default=50612,
    )

    args = ap.parse_args()

    transport = M7M5Transport(
        host=args.host,
        command_port=args.command_port,
        telemetry_port=args.telemetry_port,
        state_port=args.state_port,
    )

    samples = []

    try:
        telemetry, state = (
            transport.wait_initial(
                timeout_s=20.0
            )
        )

        print("=" * 72)
        print(
            "ICRA27 M7 TRANSPORT BOUNDARY CHECK"
        )
        print("=" * 72)

        print(
            "initial telemetry sim:",
            telemetry["sim_time_s"],
        )

        print(
            "initial state sim    :",
            state["sample_time_s"],
        )

        print()

        for name, action in ACTIONS:
            sample = transport.step(
                action,
                target_sim_dt=0.20,
                timeout_s=10.0,
            )

            samples.append(sample)

            print(
                f"[{name}] "
                f"seq={sample.command_seq}"
            )

            print(
                "  requested norm :",
                fmt(
                    sample
                    .requested_normalized
                ),
            )

            print(
                "  requested phys :",
                fmt(
                    sample
                    .requested_physical
                ),
            )

            print(
                "  applied phys   :",
                fmt(
                    sample
                    .applied_physical
                ),
            )

            print(
                "  applied norm   :",
                fmt(
                    sample
                    .applied_normalized
                ),
            )

            print(
                "  safety         :",
                sample.safety_state,
            )

            print(
                "  override       :",
                sample.override_active,
            )

            print(
                "  sim dt         :",
                f"{sample.sim_dt_s:.4f}",
            )

            print(
                "  base xyz       :",
                sample.state[
                    "base_position_world"
                ],
            )

            print()

        # --------------------------------------------------------
        # Acceptance checks
        # --------------------------------------------------------
        if len(samples) != len(ACTIONS):
            raise AssertionError(
                "missing transport samples"
            )

        seqs = [
            s.command_seq
            for s in samples
        ]

        if seqs != list(
            range(len(ACTIONS))
        ):
            raise AssertionError(
                f"unexpected command seqs {seqs}"
            )

        for sample in samples:
            if sample.applied_physical is None:
                raise AssertionError(
                    "applied command not observable"
                )

            if (
                sample.telemetry[
                    "command_seq"
                ]
                != sample.command_seq
            ):
                raise AssertionError(
                    "telemetry command_seq mismatch"
                )

            if sample.sim_dt_s < 0.15:
                raise AssertionError(
                    "high-level transition "
                    "advanced too little sim time"
                )

            if sample.safety_state not in {
                "normal",
                "watch",
                "unsafe",
            }:
                raise AssertionError(
                    "invalid safety state"
                )

        if (
            transport.bad_telemetry_packets
            != 0
        ):
            raise AssertionError(
                "bad M5 telemetry packets: "
                f"{transport.bad_telemetry_packets}"
            )

        if transport.bad_state_packets != 0:
            raise AssertionError(
                "bad M7 state packets: "
                f"{transport.bad_state_packets}"
            )

        print(
            "requested observability : PASS"
        )

        print(
            "applied observability   : PASS"
        )

        print(
            "M4 status observability : PASS"
        )

        print(
            "5 Hz sim-time boundary  : PASS"
        )

        print()
        print(
            "[ICRA27] M7 transport "
            "boundary: PASS"
        )

    finally:
        transport.close()


if __name__ == "__main__":
    main()
