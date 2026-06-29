from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, Mapping, Optional

from tracer_core.highlevel.ram_scalar_v3_udp_client import (
    RAMScalarV3UDPClient,
    RAMScalarV3UDPOutput,
)


RAM_SCALAR_V3_FEATURE_NAMES = [
    "mpc_vx",
    "mpc_yaw",
    "mpc_body_height",
    "mpc_clearance",
    "mpc_enable",
    "beta_motion",
    "beta_stability",
    "beta_energy",
    "proprio_abs_mean",
    "proprio_abs_max",
    "proprio_base_x",
    "proprio_base_y",
    "proprio_base_z",
    "odom_x",
    "odom_y",
    "odom_z",
    "odom_vx",
    "age_mpc",
    "age_beta",
    "age_proprio",
    "age_odom",
    "age_debug",
]


@dataclass
class RAMScalarV3RuntimeResult:
    ok: bool
    risk: float
    bad_prob: float
    good_prob: float
    probabilities: Dict[str, float]
    row: Dict[str, float]
    age_sec: float
    error: Optional[str] = None


def _finite_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def build_ram_scalar_v3_row(
    *,
    mpc_vx: float = 0.0,
    mpc_yaw: float = 0.0,
    mpc_body_height: float = 0.0,
    mpc_clearance: float = 0.0,
    mpc_enable: float = 0.0,
    beta_motion: float = 0.0,
    beta_stability: float = 0.0,
    beta_energy: float = 0.0,
    proprio_abs_mean: float = 0.0,
    proprio_abs_max: float = 0.0,
    proprio_base_x: float = 0.0,
    proprio_base_y: float = 0.0,
    proprio_base_z: float = 0.0,
    odom_x: float = 0.0,
    odom_y: float = 0.0,
    odom_z: float = 0.0,
    odom_vx: float = 0.0,
    age_mpc: float = 9999.0,
    age_beta: float = 9999.0,
    age_proprio: float = 9999.0,
    age_odom: float = 9999.0,
    age_debug: float = 9999.0,
    extra: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    row = {
        "mpc_vx": _finite_float(mpc_vx),
        "mpc_yaw": _finite_float(mpc_yaw),
        "mpc_body_height": _finite_float(mpc_body_height),
        "mpc_clearance": _finite_float(mpc_clearance),
        "mpc_enable": _finite_float(mpc_enable),
        "beta_motion": _finite_float(beta_motion),
        "beta_stability": _finite_float(beta_stability),
        "beta_energy": _finite_float(beta_energy),
        "proprio_abs_mean": _finite_float(proprio_abs_mean),
        "proprio_abs_max": _finite_float(proprio_abs_max),
        "proprio_base_x": _finite_float(proprio_base_x),
        "proprio_base_y": _finite_float(proprio_base_y),
        "proprio_base_z": _finite_float(proprio_base_z),
        "odom_x": _finite_float(odom_x),
        "odom_y": _finite_float(odom_y),
        "odom_z": _finite_float(odom_z),
        "odom_vx": _finite_float(odom_vx),
        "age_mpc": _finite_float(age_mpc, 9999.0),
        "age_beta": _finite_float(age_beta, 9999.0),
        "age_proprio": _finite_float(age_proprio, 9999.0),
        "age_odom": _finite_float(age_odom, 9999.0),
        "age_debug": _finite_float(age_debug, 9999.0),
    }

    if extra:
        for k, v in extra.items():
            if k in row:
                row[k] = _finite_float(v, row[k])

    return row


def build_ram_scalar_v3_row_from_mapping(
    values: Mapping[str, float],
    *,
    defaults: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    defaults = defaults or {}
    row = {}
    for name in RAM_SCALAR_V3_FEATURE_NAMES:
        default = defaults.get(name, 9999.0 if name.startswith("age_") else 0.0)
        row[name] = _finite_float(values.get(name, default), default)
    return row


class RAMScalarV3RuntimeBridge:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 50230,
        timeout_sec: float = 0.02,
        fail_safe_risk: float = 1.0,
    ):
        self.client = RAMScalarV3UDPClient(
            host=host,
            port=port,
            timeout_sec=timeout_sec,
        )
        self.fail_safe_risk = float(fail_safe_risk)
        self.last_result: RAMScalarV3RuntimeResult | None = None
        self.last_query_wall: float | None = None

    def query_row(
        self,
        row: Mapping[str, float],
        *,
        reset: bool = False,
    ) -> RAMScalarV3RuntimeResult:
        clean_row = build_ram_scalar_v3_row_from_mapping(row)
        out: RAMScalarV3UDPOutput = self.client.query_row(clean_row, reset=reset)
        self.last_query_wall = time.time()

        error = None
        if not out.ok:
            error = str(out.raw.get("error", "RAM UDP query failed"))

        result = RAMScalarV3RuntimeResult(
            ok=out.ok,
            risk=float(out.risk if out.ok else self.fail_safe_risk),
            bad_prob=float(out.bad_prob if out.ok else self.fail_safe_risk),
            good_prob=float(out.good_prob if out.ok else 0.0),
            probabilities=dict(out.probabilities),
            row=clean_row,
            age_sec=float(out.age_sec),
            error=error,
        )
        self.last_result = result
        return result

    def query_rows(
        self,
        rows,
        *,
        reset: bool = False,
    ) -> RAMScalarV3RuntimeResult:
        clean_rows = [build_ram_scalar_v3_row_from_mapping(r) for r in rows]
        out: RAMScalarV3UDPOutput = self.client.query_rows(clean_rows, reset=reset)
        self.last_query_wall = time.time()

        error = None
        if not out.ok:
            error = str(out.raw.get("error", "RAM UDP query failed"))

        result = RAMScalarV3RuntimeResult(
            ok=out.ok,
            risk=float(out.risk if out.ok else self.fail_safe_risk),
            bad_prob=float(out.bad_prob if out.ok else self.fail_safe_risk),
            good_prob=float(out.good_prob if out.ok else 0.0),
            probabilities=dict(out.probabilities),
            row=clean_rows[-1] if clean_rows else {},
            age_sec=float(out.age_sec),
            error=error,
        )
        self.last_result = result
        return result
