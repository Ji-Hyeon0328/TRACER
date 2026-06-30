#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


PROFILE_BETA = {
    "balanced":          [0.34, 0.33, 0.33],
    "motion":            [0.65, 0.20, 0.15],
    "stability":         [0.20, 0.65, 0.15],
    "energy":            [0.20, 0.20, 0.60],
    "motion_extreme":    [0.85, 0.10, 0.05],
    "stability_extreme": [0.05, 0.90, 0.05],
    "energy_extreme":    [0.05, 0.10, 0.85],
}


THETA_SPECS = {
    # dim 0
    "gait_period_scale": {"range": (0.75, 1.35), "default": 1.0},
    # dim 1
    "duty_factor_delta": {"range": (-0.12, 0.16), "default": 0.0},
    # dim 2
    "step_length_scale": {"range": (0.35, 1.25), "default": 1.0},
    # dim 3
    "stance_width_delta": {"range": (-0.04, 0.06), "default": 0.0},
    # dim 4
    "body_height_delta": {"range": (-0.04, 0.06), "default": 0.0},
    # dim 5
    "clearance_delta": {"range": (0.00, 0.09), "default": 0.0},
    # dim 6
    "impedance_scale": {"range": (0.70, 1.60), "default": 1.0},
    # dim 7
    "residual_scale": {"range": (0.00, 1.50), "default": 0.5},
}


def normalize_beta(beta: list[float]) -> list[float]:
    vals = [float(x) for x in beta]
    s = sum(vals)
    if not math.isfinite(s) or s <= 1e-9:
        return [1.0 / 3.0] * 3
    return [x / s for x in vals]


def l1(a: list[float], b: list[float]) -> float:
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b))


def theta_key(theta: list[float], ndigits: int = 5) -> tuple[float, ...]:
    return tuple(round(float(x), ndigits) for x in theta)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_path(path_s: str) -> Path:
    p = Path(path_s).expanduser()
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def selector_score(
    r: dict[str, Any],
    query_beta: list[float],
    beta_dist_weight: float,
    std_weight: float,
) -> float:
    pb = normalize_beta(r.get("profile_beta", [0.34, 0.33, 0.33]))
    beta_dist = l1(query_beta, pb)

    return (
        float(r["R_profile_gated_mean"])
        - beta_dist_weight * beta_dist
        - std_weight * float(r.get("R_profile_gated_std", 0.0))
        + 0.03 * float(r.get("feasible_frac", 1.0))
        + 0.05 * float(r.get("distance_mean", 0.0))
        + 0.05 * float(r.get("zmin_mean", 0.0))
    )


def select_from_bank(
    rows: list[dict[str, Any]],
    query_beta: list[float],
    beta_dist_weight: float = 0.35,
    std_weight: float = 0.35,
) -> dict[str, Any]:
    scored = []
    for r in rows:
        if "theta_norm" not in r or r["theta_norm"] is None:
            continue
        r2 = dict(r)
        r2["selector_score"] = selector_score(
            r2,
            query_beta=query_beta,
            beta_dist_weight=beta_dist_weight,
            std_weight=std_weight,
        )
        scored.append(r2)

    if not scored:
        raise RuntimeError("No selectable theta rows in safe bank.")

    # Deduplicate by theta. Keep highest selector score.
    by_theta: dict[tuple[float, ...], dict[str, Any]] = {}
    for r in scored:
        key = theta_key([float(x) for x in r["theta_norm"]])
        old = by_theta.get(key)
        if old is None or r["selector_score"] > old["selector_score"]:
            by_theta[key] = r

    unique = list(by_theta.values())
    unique.sort(key=lambda r: r["selector_score"], reverse=True)
    return unique[0]


def denorm_with_default(a: float, lo: float, default: float, hi: float) -> float:
    a = max(-1.0, min(1.0, float(a)))
    if a >= 0.0:
        return default + a * (hi - default)
    return default + (-a) * (lo - default)


def theta_norm_to_physical(theta: list[float]) -> dict[str, float]:
    if len(theta) != 8:
        raise ValueError(f"theta must have length 8, got {len(theta)}")

    names = [
        "gait_period_scale",
        "duty_factor_delta",
        "step_length_scale",
        "stance_width_delta",
        "body_height_delta",
        "clearance_delta",
        "impedance_scale",
        "residual_scale",
    ]

    out: dict[str, float] = {}
    for name, a in zip(names, theta):
        spec = THETA_SPECS[name]
        lo, hi = spec["range"]
        default = spec["default"]
        out[name] = denorm_with_default(float(a), float(lo), float(default), float(hi))

    return out


def theta_to_mpc_ref(theta: list[float], counter: float = 0.0) -> list[float]:
    # Keep this mapping aligned with theta_direct_mapper_v0 for the current ROS1
    # A1-QP-MPC bridge, which consumes:
    # [counter, vx, yaw_rate, body_height, swing_clearance, enable].
    phy = theta_norm_to_physical(theta)

    nominal_vx = 0.06
    nominal_yaw_rate = 0.0
    nominal_body_height = 0.300
    nominal_clearance = 0.035

    vx = nominal_vx * float(phy["step_length_scale"])
    yaw_rate = nominal_yaw_rate
    body_height = nominal_body_height + float(phy["body_height_delta"])
    clearance = nominal_clearance + float(phy["clearance_delta"])

    # Runtime clamps for current bridge.
    vx = max(-0.05, min(0.16, vx))
    body_height = max(0.24, min(0.36, body_height))
    clearance = max(0.02, min(0.12, clearance))

    return [float(counter), float(vx), float(yaw_rate), float(body_height), float(clearance), 1.0]


class TracerSafeBankPolicyNode(Node):
    def __init__(self):
        super().__init__("tracer_safe_bank_policy_node_v0")

        self.declare_parameter("safe_bank_path", "artifacts/theta_safe_bank_v1/theta_safe_bank_v1.jsonl")
        self.declare_parameter("profile", "balanced")
        self.declare_parameter("beta_motion", 0.34)
        self.declare_parameter("beta_stability", 0.33)
        self.declare_parameter("beta_energy", 0.33)
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("beta_dist_weight", 0.35)
        self.declare_parameter("std_weight", 0.35)
        self.declare_parameter("mpc_ref_topic", "/tracer/mpc_reference")
        self.declare_parameter("theta_topic", "/tracer/selected_theta")
        self.declare_parameter("enable_beta_sub", True)
        self.declare_parameter("beta_topic", "/tracer/objective_beta")
        self.declare_parameter("runtime_phase_topic", "/tracer/runtime_phase")
        self.declare_parameter("hold_vx", 0.0)
        self.declare_parameter("hold_yaw_rate", 0.0)
        self.declare_parameter("hold_body_height", 0.300)
        self.declare_parameter("hold_clearance", 0.035)
        self.declare_parameter("hold_enable", 1.0)

        self.safe_bank_path = resolve_path(str(self.get_parameter("safe_bank_path").value))
        self.profile = str(self.get_parameter("profile").value)
        self.beta_dist_weight = float(self.get_parameter("beta_dist_weight").value)
        self.std_weight = float(self.get_parameter("std_weight").value)

        self.rows = read_jsonl(self.safe_bank_path)

        self.runtime_phase = "policy"

        self.current_beta = self._initial_beta()
        self.selected = select_from_bank(
            self.rows,
            query_beta=self.current_beta,
            beta_dist_weight=self.beta_dist_weight,
            std_weight=self.std_weight,
        )
        self.theta = [float(x) for x in self.selected["theta_norm"]]

        self.counter = 0.0

        self.pub_ref = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("mpc_ref_topic").value),
            10,
        )
        self.pub_theta = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("theta_topic").value),
            10,
        )

        if bool(self.get_parameter("enable_beta_sub").value):
            self.create_subscription(
                Float64MultiArray,
                str(self.get_parameter("beta_topic").value),
                self._on_beta,
                10,
            )

        self.create_subscription(
            String,
            str(self.get_parameter("runtime_phase_topic").value),
            self._on_runtime_phase,
            10,
        )

        hz = max(1.0, float(self.get_parameter("publish_hz").value))
        self.timer = self.create_timer(1.0 / hz, self._on_timer)

        self._log_selected("initial")

    def _initial_beta(self) -> list[float]:
        if self.profile in PROFILE_BETA:
            return normalize_beta(PROFILE_BETA[self.profile])

        return normalize_beta([
            float(self.get_parameter("beta_motion").value),
            float(self.get_parameter("beta_stability").value),
            float(self.get_parameter("beta_energy").value),
        ])

    def _log_selected(self, reason: str):
        ref = theta_to_mpc_ref(self.theta, counter=0.0)
        self.get_logger().info(
            "[TRACER] safe-bank selected "
            f"reason={reason} "
            f"profile={self.profile} "
            f"beta={self.current_beta} "
            f"src={self.selected.get('source_tag')} "
            f"preset={self.selected.get('preset')} "
            f"score={float(self.selected.get('selector_score', 0.0)):.4f} "
            f"Rprof={float(self.selected.get('R_profile_gated_mean', 0.0)):.4f} "
            f"feasible={self.selected.get('feasible_count')}/{self.selected.get('n')} "
            f"theta={self.theta} "
            f"ref={ref}"
        )

    def _on_runtime_phase(self, msg: String):
        phase = str(msg.data).strip().lower()
        if phase not in {"hold", "stand", "reset", "policy"}:
            self.get_logger().warn(f"[TRACER] ignoring unknown runtime_phase={phase}")
            return

        if phase in {"stand", "reset"}:
            phase = "hold"

        if phase != self.runtime_phase:
            self.runtime_phase = phase
            self.get_logger().info(f"[TRACER] runtime_phase -> {self.runtime_phase}")

    def _on_beta(self, msg: Float64MultiArray):
        if len(msg.data) < 3:
            return

        beta = normalize_beta([float(msg.data[0]), float(msg.data[1]), float(msg.data[2])])
        if l1(beta, self.current_beta) < 1e-4:
            return

        self.current_beta = beta
        self.profile = "dynamic_beta"
        self.selected = select_from_bank(
            self.rows,
            query_beta=self.current_beta,
            beta_dist_weight=self.beta_dist_weight,
            std_weight=self.std_weight,
        )
        self.theta = [float(x) for x in self.selected["theta_norm"]]
        self._log_selected("beta_update")

    def _on_timer(self):
        if self.runtime_phase == "hold":
            ref = [
                float(self.counter),
                float(self.get_parameter("hold_vx").value),
                float(self.get_parameter("hold_yaw_rate").value),
                float(self.get_parameter("hold_body_height").value),
                float(self.get_parameter("hold_clearance").value),
                float(self.get_parameter("hold_enable").value),
            ]
        else:
            ref = theta_to_mpc_ref(self.theta, counter=self.counter)

        ref_msg = Float64MultiArray()
        ref_msg.data = ref
        self.pub_ref.publish(ref_msg)

        theta_msg = Float64MultiArray()
        theta_msg.data = self.theta
        self.pub_theta.publish(theta_msg)

        self.counter += 1.0


def main():
    rclpy.init()
    node = TracerSafeBankPolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
