#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.lowlevel.pympc_udp_protocol import (
    COMMAND_SCHEMA,
    TELEMETRY_SCHEMA,
    decode_command,
    decode_telemetry,
    encode_command,
    encode_telemetry,
    make_command,
    make_telemetry_payload,
)


def must_fail(fn):
    try:
        fn()
    except (ValueError, TypeError):
        return

    raise AssertionError(
        "expected failure did not occur"
    )


def main():
    command = make_command(
        seq=7,
        stamp=123.0,
        label="ros2",
        vx=0.20,
        yaw_rate=0.10,
        body_height=0.30,
        swing_clearance=0.06,
        gait_period=1.0 / 1.4,
        duty_factor=0.65,
    )

    decoded = decode_command(
        encode_command(command)
    )

    assert decoded == command
    assert decoded.values == command.values

    telemetry = make_telemetry_payload(
        seq=11,
        command_seq=7,
        requested_label="ros2_seq_7",
        selected_label="ros2_seq_7",
        safety_state="normal",
        override_active=False,
        structural_commit=False,
        sim_time_s=3.216,
        structural_commit_count=1,
        command_fresh=True,
        command_age_s=0.012,
        applied=command,
    )

    decoded_telem = decode_telemetry(
        encode_telemetry(telemetry)
    )

    assert (
        decoded_telem["schema"]
        == TELEMETRY_SCHEMA
    )

    assert (
        decoded_telem["command_seq"]
        == 7
    )

    assert (
        decoded_telem["applied"]["vx"]
        == 0.20
    )

    assert (
        decoded_telem["sim_time_s"]
        == 3.216
    )

    assert (
        decoded_telem[
            "structural_commit_count"
        ]
        == 1
    )

    bad_sim_time = dict(telemetry)
    bad_sim_time["sim_time_s"] = -1.0

    must_fail(
        lambda: decode_telemetry(
            json.dumps(
                bad_sim_time
            ).encode("utf-8")
        )
    )

    bad_commit_count = dict(telemetry)
    bad_commit_count[
        "structural_commit_count"
    ] = -1

    must_fail(
        lambda: decode_telemetry(
            json.dumps(
                bad_commit_count
            ).encode("utf-8")
        )
    )

    bad_schema = {
        **command.as_payload(),
        "schema": "wrong.schema",
    }

    must_fail(
        lambda: decode_command(
            json.dumps(
                bad_schema
            ).encode("utf-8")
        )
    )

    zero_period = (
        command.as_payload()
    )

    zero_period["command"][
        "gait_period"
    ] = 0.0

    must_fail(
        lambda: decode_command(
            json.dumps(
                zero_period
            ).encode("utf-8")
        )
    )

    nan_packet = (
        command.as_payload()
    )

    nan_packet["command"]["vx"] = (
        float("nan")
    )

    must_fail(
        lambda: decode_command(
            json.dumps(
                nan_packet
            ).encode("utf-8")
        )
    )

    assert COMMAND_SCHEMA.endswith(".v1")
    assert TELEMETRY_SCHEMA.endswith(".v1")

    print(
        "PyMPC UDP protocol checker: PASS"
    )
    print(
        "command schema:",
        COMMAND_SCHEMA,
    )
    print(
        "telemetry schema:",
        TELEMETRY_SCHEMA,
    )
    print(
        "command fields:"
        " vx yaw_rate body_height"
        " swing_clearance gait_period"
        " duty_factor"
    )


if __name__ == "__main__":
    main()
