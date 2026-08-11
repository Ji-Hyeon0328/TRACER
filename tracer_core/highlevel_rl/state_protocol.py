from __future__ import annotations

import json
from math import isfinite
from typing import Any, Mapping


SCHEMA = "icra27_m7_pympc_state_v0"


_VECTOR_FIELDS = {
    "base_position_world": 3,
    "base_rpy": 3,
    "base_linear_velocity_world": 3,
    "base_linear_velocity_body_yaw": 3,
    "base_angular_velocity_base": 3,
}


def _finite_float(name: str, x: Any) -> float:
    value = float(x)

    if not isfinite(value):
        raise ValueError(
            f"{name} must be finite, got {x!r}"
        )

    return value


def validate_state_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(payload)

    if out.get("schema") != SCHEMA:
        raise ValueError(
            f"unexpected schema={out.get('schema')!r}"
        )

    out["seq"] = int(out["seq"])
    out["episode_index"] = int(
        out["episode_index"]
    )

    out["sample_time_s"] = _finite_float(
        "sample_time_s",
        out["sample_time_s"],
    )

    out["lowlevel_dt_s"] = _finite_float(
        "lowlevel_dt_s",
        out["lowlevel_dt_s"],
    )

    if out["lowlevel_dt_s"] <= 0.0:
        raise ValueError(
            "lowlevel_dt_s must be positive"
        )

    for name, size in _VECTOR_FIELDS.items():
        values = [
            _finite_float(
                f"{name}[{i}]",
                x,
            )
            for i, x in enumerate(out[name])
        ]

        if len(values) != size:
            raise ValueError(
                f"{name} must have length {size}, "
                f"got {len(values)}"
            )

        out[name] = values

    out["terminated"] = bool(
        out["terminated"]
    )

    out["truncated"] = bool(
        out["truncated"]
    )

    out["sample_phase"] = str(
        out["sample_phase"]
    )

    if (
        out["sample_phase"]
        != "controller_input_pre_env_step"
    ):
        raise ValueError(
            "unexpected sample_phase="
            f"{out['sample_phase']!r}"
        )

    native_reward = out.get(
        "native_reward"
    )

    if native_reward is not None:
        out["native_reward"] = (
            _finite_float(
                "native_reward",
                native_reward,
            )
        )

    mechanical_energy = out.get(
        "mechanical_energy"
    )

    if mechanical_energy is not None:
        if not isinstance(
            mechanical_energy,
            Mapping,
        ):
            raise ValueError(
                "mechanical_energy must "
                "be a mapping"
            )

        required_energy_fields = (
            "samples",
            "elapsed_s",
            "commanded_signed_j",
            "commanded_abs_j",
            "commanded_positive_j",
            "applied_signed_j",
            "applied_abs_j",
            "applied_positive_j",
        )

        missing_energy_fields = [
            name
            for name in required_energy_fields
            if name not in mechanical_energy
        ]

        if missing_energy_fields:
            raise ValueError(
                "mechanical_energy missing fields: "
                f"{missing_energy_fields}"
            )

        energy = {
            "samples":
                int(
                    mechanical_energy[
                        "samples"
                    ]
                ),
        }

        if energy["samples"] < 0:
            raise ValueError(
                "mechanical_energy.samples "
                "must be >= 0"
            )

        for name in (
            "elapsed_s",
            "commanded_signed_j",
            "commanded_abs_j",
            "commanded_positive_j",
            "applied_signed_j",
            "applied_abs_j",
            "applied_positive_j",
        ):
            energy[name] = _finite_float(
                f"mechanical_energy.{name}",
                mechanical_energy[name],
            )

        for name in (
            "elapsed_s",
            "commanded_abs_j",
            "commanded_positive_j",
            "applied_abs_j",
            "applied_positive_j",
        ):
            if energy[name] < 0.0:
                raise ValueError(
                    f"mechanical_energy.{name} "
                    "must be >= 0"
                )

        if (
            energy["commanded_positive_j"]
            > energy["commanded_abs_j"]
            + 1e-9
        ):
            raise ValueError(
                "commanded positive work "
                "exceeds absolute work"
            )

        if (
            energy["applied_positive_j"]
            > energy["applied_abs_j"]
            + 1e-9
        ):
            raise ValueError(
                "applied positive work "
                "exceeds absolute work"
            )

        out["mechanical_energy"] = energy

        out["energy_sample_time_s"] = (
            _finite_float(
                "energy_sample_time_s",
                out["energy_sample_time_s"],
            )
        )

    return out


def encode_state_payload(
    payload: Mapping[str, Any],
) -> bytes:
    valid = validate_state_payload(
        payload
    )

    return json.dumps(
        valid,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def decode_state_payload(
    raw: bytes,
) -> dict[str, Any]:
    obj = json.loads(
        raw.decode("utf-8")
    )

    if not isinstance(obj, dict):
        raise ValueError(
            "state packet must decode to object"
        )

    return validate_state_payload(obj)
