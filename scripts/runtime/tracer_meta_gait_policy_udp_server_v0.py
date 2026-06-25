#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracer_core.highlevel.meta_gait import (  # noqa: E402
    HighLevelPolicyInput,
    ObjectiveWeights,
    RamSignal,
    tuple_from_sequence,
)
from tracer_core.highlevel.meta_gait_dataset import TARGET_NAMES  # noqa: E402
from tracer_core.highlevel.meta_gait_policy import make_meta_gait_policy  # noqa: E402


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _as_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    return str(x)


def _mapping(x: Any) -> Mapping[str, Any]:
    return x if isinstance(x, Mapping) else {}


def _objective_from_request(req: Mapping[str, Any]) -> ObjectiveWeights:
    beta = req.get("beta", None)

    if isinstance(beta, Mapping):
        return ObjectiveWeights.from_mapping(beta).normalized()

    if isinstance(beta, (list, tuple)) and len(beta) >= 3:
        return ObjectiveWeights(
            motion=_as_float(beta[0], 1.0 / 3.0),
            stability=_as_float(beta[1], 1.0 / 3.0),
            energy=_as_float(beta[2], 1.0 / 3.0),
        ).normalized()

    return ObjectiveWeights().normalized()


def _ram_from_request(req: Mapping[str, Any]) -> RamSignal:
    ram = _mapping(req.get("ram", {}))

    return RamSignal(
        rho=tuple_from_sequence(ram.get("rho", req.get("rho", []))),
        sigma=_as_float(ram.get("sigma", req.get("sigma", 0.0)), 0.0),
        ram_level=_as_str(ram.get("ram_level", req.get("ram_level", "unknown")), "unknown"),
        control_risk=_as_float(ram.get("control_risk", req.get("control_risk", 0.0)), 0.0),
        fallen_prob=_as_float(ram.get("fallen_prob", req.get("fallen_prob", 0.0)), 0.0),
        recovery_prob=_as_float(ram.get("recovery_prob", req.get("recovery_prob", 0.0)), 0.0),
    )


def _policy_input_from_request(req: Mapping[str, Any]) -> HighLevelPolicyInput:
    return HighLevelPolicyInput(
        context=tuple_from_sequence(req.get("context", [])),
        ram=_ram_from_request(req),
        beta=_objective_from_request(req),
        gait_mode=_as_str(req.get("gait_mode", req.get("mode", "unknown")), "unknown"),
        robot_state=tuple_from_sequence(req.get("robot_state", [])),
        goal=tuple_from_sequence(req.get("goal", [])),
    )


def _meta_to_response(meta, *, request_id: Any = None) -> dict[str, Any]:
    target_vector = [
        float(meta.vx),
        float(meta.yaw_rate),
        float(meta.body_height),
        float(meta.swing_clearance),
        float(meta.enable),
        float(meta.gait_period),
        float(meta.duty_factor),
        float(meta.step_length),
        float(meta.stance_width),
        float(meta.impedance_scale),
        float(meta.residual_gain_scale),
        float(meta.risk_scale),
    ]

    return {
        "ok": True,
        "request_id": request_id,
        "target_names": list(TARGET_NAMES),
        "target_vector": target_vector,
        "meta_gait": {
            "vx": float(meta.vx),
            "yaw_rate": float(meta.yaw_rate),
            "body_height": float(meta.body_height),
            "swing_clearance": float(meta.swing_clearance),
            "enable": float(meta.enable),
            "gait_period": float(meta.gait_period),
            "duty_factor": float(meta.duty_factor),
            "step_length": float(meta.step_length),
            "stance_width": float(meta.stance_width),
            "impedance_scale": float(meta.impedance_scale),
            "residual_gain_scale": float(meta.residual_gain_scale),
            "risk_scale": float(meta.risk_scale),
            "source_mode": str(meta.source_mode),
            "source_reason": str(meta.source_reason),
            "extras": dict(meta.extras),
        },
        "stamp_wall": time.time(),
    }


def _error_response(exc: Exception, *, request_id: Any = None) -> dict[str, Any]:
    return {
        "ok": False,
        "request_id": request_id,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "stamp_wall": time.time(),
    }


def main() -> None:
    host = os.environ.get("TRACER_META_GAIT_POLICY_UDP_HOST", "127.0.0.1")
    port = int(os.environ.get("TRACER_META_GAIT_POLICY_UDP_PORT", "50310"))
    policy_kind = os.environ.get("TRACER_META_GAIT_POLICY_KIND", "learned")
    model_path = os.environ.get(
        "TRACER_META_GAIT_POLICY_MODEL",
        str(ROOT / "artifacts/meta_gait_policy_v0/model.pt"),
    )
    max_bytes = int(os.environ.get("TRACER_META_GAIT_POLICY_UDP_MAX_BYTES", "65535"))

    print("[TRACER] meta-gait policy UDP server v0")
    print(f"[TRACER] root:   {ROOT}")
    print(f"[TRACER] bind:   {host}:{port}")
    print(f"[TRACER] kind:   {policy_kind}")
    print(f"[TRACER] model:  {model_path if policy_kind in {'learned', 'torch'} else '<none>'}")
    print("[TRACER] request JSON fields:")
    print("  gait_mode, beta, ram/rho/sigma/context/robot_state/goal")
    print("  or {'ping': true}")

    policy = make_meta_gait_policy(
        policy_kind,
        model_path if policy_kind in {"learned", "torch"} else None,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print("[TRACER] listening...")

    while True:
        data, addr = sock.recvfrom(max_bytes)

        request_id = None
        try:
            req = json.loads(data.decode("utf-8"))
            if not isinstance(req, Mapping):
                raise ValueError("request must be a JSON object")

            request_id = req.get("request_id", None)

            if req.get("ping", False):
                resp = {
                    "ok": True,
                    "request_id": request_id,
                    "pong": True,
                    "policy_kind": policy_kind,
                    "model_path": model_path,
                    "stamp_wall": time.time(),
                }
            else:
                inp = _policy_input_from_request(req)
                meta = policy.predict(inp)
                resp = _meta_to_response(meta, request_id=request_id)

        except Exception as exc:
            resp = _error_response(exc, request_id=request_id)

        sock.sendto(json.dumps(resp).encode("utf-8"), addr)


if __name__ == "__main__":
    main()
