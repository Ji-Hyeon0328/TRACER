#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.highlevel_rl.state_protocol import (
    decode_state_payload,
)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--host",
        default="127.0.0.1",
    )

    ap.add_argument(
        "--port",
        type=int,
        default=50512,
    )

    ap.add_argument(
        "--timeout",
        type=float,
        default=15.0,
    )

    ap.add_argument(
        "--expect-min-packets",
        type=int,
        default=20,
    )

    args = ap.parse_args()

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1,
    )

    sock.bind(
        (
            args.host,
            args.port,
        )
    )

    sock.settimeout(0.25)

    packets = []
    deadline = (
        time.monotonic()
        + args.timeout
    )

    try:
        while (
            time.monotonic()
            < deadline
            and len(packets)
            < args.expect_min_packets
        ):
            try:
                raw, _ = sock.recvfrom(
                    65535
                )

            except socket.timeout:
                continue

            payload = decode_state_payload(
                raw
            )

            packets.append(
                payload
            )

    finally:
        sock.close()

    if (
        len(packets)
        < args.expect_min_packets
    ):
        raise SystemExit(
            "FAIL: received only "
            f"{len(packets)} state packets, "
            f"expected >= "
            f"{args.expect_min_packets}"
        )

    seq = [
        p["seq"]
        for p in packets
    ]

    if any(
        b <= a
        for a, b in zip(
            seq,
            seq[1:],
        )
    ):
        raise SystemExit(
            f"FAIL: non-monotonic seq={seq}"
        )

    sim_t = [
        p["sample_time_s"]
        for p in packets
    ]

    if any(
        b < a
        for a, b in zip(
            sim_t,
            sim_t[1:],
        )
    ):
        raise SystemExit(
            "FAIL: simulation time went backward"
        )

    first = packets[0]
    last = packets[-1]

    print("=" * 72)
    print(
        "ICRA27 M7 READ-ONLY STATE TAP CHECK"
    )
    print("=" * 72)

    print(
        "packets received :",
        len(packets),
    )

    print(
        "seq range        :",
        seq[0],
        "->",
        seq[-1],
    )

    print(
        "sim time range   :",
        f"{sim_t[0]:.4f}",
        "->",
        f"{sim_t[-1]:.4f}",
    )

    print(
        "sample phase     :",
        last["sample_phase"],
    )

    print(
        "base xyz         :",
        last["base_position_world"],
    )

    print(
        "base rpy         :",
        last["base_rpy"],
    )

    print(
        "world velocity   :",
        last["base_linear_velocity_world"],
    )

    print(
        "body-yaw velocity:",
        last[
            "base_linear_velocity_body_yaw"
        ],
    )

    print(
        "base angular vel :",
        last[
            "base_angular_velocity_base"
        ],
    )

    print(
        "terminated       :",
        last["terminated"],
    )

    print(
        "truncated        :",
        last["truncated"],
    )

    print()
    print(
        "[ICRA27] M7 read-only "
        "MuJoCo state tap: PASS"
    )


if __name__ == "__main__":
    main()
