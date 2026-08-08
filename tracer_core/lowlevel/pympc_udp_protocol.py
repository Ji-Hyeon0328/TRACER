from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any, Mapping


COMMAND_SCHEMA = "tracer.meta_gait.command.v1"
TELEMETRY_SCHEMA = "tracer.pympc.telemetry.v1"

COMMAND_FIELDS = (
    "vx",
    "yaw_rate",
    "body_height",
    "swing_clearance",
    "gait_period",
    "duty_factor",
)


def _finite_float(
    value: Any,
    *,
    name: str,
) -> float:
    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"{name} must be finite"
        )

    return result


def _nonnegative_int(
    value: Any,
    *,
    name: str,
) -> int:
    result = int(value)

    if result < 0:
        raise ValueError(
            f"{name} must be >= 0"
        )

    return result


@dataclass(frozen=True)
class MetaGaitUdpCommand:
    seq: int
    stamp: float
    label: str

    vx: float
    yaw_rate: float
    body_height: float
    swing_clearance: float
    gait_period: float
    duty_factor: float

    @property
    def values(self) -> tuple[float, ...]:
        return (
            self.vx,
            self.yaw_rate,
            self.body_height,
            self.swing_clearance,
            self.gait_period,
            self.duty_factor,
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema": COMMAND_SCHEMA,
            "seq": self.seq,
            "stamp": self.stamp,
            "label": self.label,
            "command": {
                "vx": self.vx,
                "yaw_rate": self.yaw_rate,
                "body_height":
                    self.body_height,
                "swing_clearance":
                    self.swing_clearance,
                "gait_period":
                    self.gait_period,
                "duty_factor":
                    self.duty_factor,
            },
        }


def make_command(
    *,
    seq: int,
    vx: float,
    yaw_rate: float,
    body_height: float,
    swing_clearance: float,
    gait_period: float,
    duty_factor: float,
    label: str = "ros2",
    stamp: float | None = None,
) -> MetaGaitUdpCommand:
    if stamp is None:
        stamp = time.time()

    gait_period_f = _finite_float(
        gait_period,
        name="gait_period",
    )

    if gait_period_f <= 0.0:
        raise ValueError(
            "gait_period must be > 0"
        )

    return MetaGaitUdpCommand(
        seq=_nonnegative_int(
            seq,
            name="seq",
        ),
        stamp=_finite_float(
            stamp,
            name="stamp",
        ),
        label=str(label),
        vx=_finite_float(
            vx,
            name="vx",
        ),
        yaw_rate=_finite_float(
            yaw_rate,
            name="yaw_rate",
        ),
        body_height=_finite_float(
            body_height,
            name="body_height",
        ),
        swing_clearance=_finite_float(
            swing_clearance,
            name="swing_clearance",
        ),
        gait_period=gait_period_f,
        duty_factor=_finite_float(
            duty_factor,
            name="duty_factor",
        ),
    )


def encode_command(
    command: MetaGaitUdpCommand,
) -> bytes:
    return json.dumps(
        command.as_payload(),
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def decode_command(
    raw: bytes,
) -> MetaGaitUdpCommand:
    payload = json.loads(
        raw.decode("utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "command packet must be an object"
        )

    if payload.get("schema") != COMMAND_SCHEMA:
        raise ValueError(
            "unexpected command schema: "
            f"{payload.get('schema')!r}"
        )

    command = payload.get("command")

    if not isinstance(command, dict):
        raise ValueError(
            "command field must be an object"
        )

    missing = [
        field
        for field in COMMAND_FIELDS
        if field not in command
    ]

    if missing:
        raise ValueError(
            f"missing command fields: {missing}"
        )

    return make_command(
        seq=payload.get("seq"),
        stamp=payload.get("stamp"),
        label=payload.get(
            "label",
            "ros2",
        ),
        vx=command["vx"],
        yaw_rate=command["yaw_rate"],
        body_height=command["body_height"],
        swing_clearance=(
            command["swing_clearance"]
        ),
        gait_period=command["gait_period"],
        duty_factor=command["duty_factor"],
    )


def command_mapping(
    command: Any,
) -> dict[str, float]:
    result = {
        "vx":
            _finite_float(
                command.vx,
                name="vx",
            ),

        "yaw_rate":
            _finite_float(
                command.yaw_rate,
                name="yaw_rate",
            ),

        "body_height":
            _finite_float(
                command.body_height,
                name="body_height",
            ),

        "swing_clearance":
            _finite_float(
                command.swing_clearance,
                name="swing_clearance",
            ),

        "gait_period":
            _finite_float(
                command.gait_period,
                name="gait_period",
            ),

        "duty_factor":
            _finite_float(
                command.duty_factor,
                name="duty_factor",
            ),
    }

    if result["gait_period"] <= 0.0:
        raise ValueError(
            "gait_period must be > 0"
        )

    return result


def make_telemetry_payload(
    *,
    seq: int,
    command_seq: int | None,
    requested_label: str | None,
    selected_label: str | None,
    safety_state: str,
    override_active: bool,
    structural_commit: bool,
    sim_time_s: float,
    structural_commit_count: int,
    command_fresh: bool,
    command_age_s: float | None,
    applied: Any | None,
    override_reasons=(),
    stamp: float | None = None,
) -> dict[str, Any]:
    if stamp is None:
        stamp = time.time()

    safety = str(safety_state).lower()

    if safety not in {
        "normal",
        "watch",
        "unsafe",
    }:
        raise ValueError(
            f"invalid safety state: {safety!r}"
        )

    if command_age_s is not None:
        command_age_s = _finite_float(
            command_age_s,
            name="command_age_s",
        )

        if command_age_s < 0.0:
            command_age_s = 0.0

    sim_time_s = _finite_float(
        sim_time_s,
        name="sim_time_s",
    )

    if sim_time_s < 0.0:
        raise ValueError(
            "sim_time_s must be >= 0"
        )

    structural_commit_count = (
        _nonnegative_int(
            structural_commit_count,
            name="structural_commit_count",
        )
    )

    return {
        "schema": TELEMETRY_SCHEMA,
        "seq": _nonnegative_int(
            seq,
            name="seq",
        ),
        "stamp": _finite_float(
            stamp,
            name="stamp",
        ),
        "command_seq": (
            None
            if command_seq is None
            else _nonnegative_int(
                command_seq,
                name="command_seq",
            )
        ),
        "requested_label":
            requested_label,
        "selected_label":
            selected_label,
        "safety_state":
            safety,
        "override_active":
            bool(override_active),
        "override_reasons":
            [
                str(x)
                for x in override_reasons
            ],
        "structural_commit":
            bool(structural_commit),
        "sim_time_s":
            sim_time_s,
        "structural_commit_count":
            structural_commit_count,
        "command_fresh":
            bool(command_fresh),
        "command_age_s":
            command_age_s,
        "applied": (
            None
            if applied is None
            else command_mapping(applied)
        ),
    }


def encode_telemetry(
    payload: Mapping[str, Any],
) -> bytes:
    if (
        payload.get("schema")
        != TELEMETRY_SCHEMA
    ):
        raise ValueError(
            "telemetry payload has "
            "unexpected schema"
        )

    return json.dumps(
        dict(payload),
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def decode_telemetry(
    raw: bytes,
) -> dict[str, Any]:
    payload = json.loads(
        raw.decode("utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "telemetry packet "
            "must be an object"
        )

    if (
        payload.get("schema")
        != TELEMETRY_SCHEMA
    ):
        raise ValueError(
            "unexpected telemetry schema: "
            f"{payload.get('schema')!r}"
        )

    _nonnegative_int(
        payload.get("seq"),
        name="seq",
    )

    if "sim_time_s" in payload:
        sim_time_s = _finite_float(
            payload["sim_time_s"],
            name="sim_time_s",
        )

        if sim_time_s < 0.0:
            raise ValueError(
                "sim_time_s must be >= 0"
            )

    if "structural_commit_count" in payload:
        _nonnegative_int(
            payload[
                "structural_commit_count"
            ],
            name="structural_commit_count",
        )

    safety = str(
        payload.get("safety_state")
    ).lower()

    if safety not in {
        "normal",
        "watch",
        "unsafe",
    }:
        raise ValueError(
            f"invalid safety state: {safety!r}"
        )

    applied = payload.get("applied")

    if applied is not None:
        if not isinstance(applied, dict):
            raise ValueError(
                "applied must be object or null"
            )

        for field in COMMAND_FIELDS:
            if field not in applied:
                raise ValueError(
                    "missing applied field: "
                    f"{field}"
                )

            _finite_float(
                applied[field],
                name=f"applied.{field}",
            )

        if float(
            applied["gait_period"]
        ) <= 0.0:
            raise ValueError(
                "applied.gait_period "
                "must be > 0"
            )

    return payload
