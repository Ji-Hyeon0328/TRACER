#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


PROFILE_BETA = {
    "balanced": [0.34, 0.33, 0.33],
    "motion": [0.65, 0.20, 0.15],
    "stability": [0.20, 0.65, 0.15],
    "energy": [0.20, 0.20, 0.60],
    "motion_extreme": [0.85, 0.10, 0.05],
    "stability_extreme": [0.05, 0.90, 0.05],
    "energy_extreme": [0.05, 0.10, 0.85],
}


def infer_profile_from_beta(beta):
    best_name = "balanced"
    best_dist = float("inf")
    for name, ref in PROFILE_BETA.items():
        dist = sum((float(a) - float(b)) ** 2 for a, b in zip(beta, ref))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name, best_dist


def denorm_with_default(a: float, lo: float, default: float, hi: float) -> float:
    a = max(-1.0, min(1.0, float(a)))
    if a >= 0.0:
        return default + a * (hi - default)
    return default + a * (default - lo)


def theta_to_mpc_ref(theta, counter: float):
    # 8D theta:
    # 0 period, 1 duty, 2 step_length_scale, 3 stance_width,
    # 4 body_height_delta, 5 clearance_delta, 6 impedance, 7 residual
    t = [float(x) for x in theta]
    while len(t) < 8:
        t.append(0.0)

    nominal_vx = 0.06
    nominal_body_height = 0.300
    nominal_clearance = 0.035

    step_length_scale = denorm_with_default(t[2], 0.35, 1.00, 1.25)
    body_height_delta = denorm_with_default(t[4], -0.04, 0.0, 0.06)
    clearance_delta = denorm_with_default(t[5], 0.0, 0.0, 0.09)

    vx = nominal_vx * step_length_scale
    yaw_rate = 0.0
    body_height = nominal_body_height + body_height_delta
    clearance = nominal_clearance + clearance_delta
    enable = 1.0

    return [
        float(counter),
        float(vx),
        float(yaw_rate),
        float(body_height),
        float(clearance),
        float(enable),
    ]


class RuntimeValidatedPolicyNode(Node):
    def __init__(self):
        super().__init__("tracer_runtime_validated_policy_node_v0")

        self.declare_parameter(
            "selector_path",
            "configs/runtime/tracer_runtime_validated_theta_selector_v0.json",
        )
        self.declare_parameter("profile", "balanced")
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("enable_beta_sub", True)
        self.declare_parameter("beta_topic", "/tracer/objective_beta")
        self.declare_parameter("runtime_phase_topic", "/tracer/runtime_phase")
        self.declare_parameter("selected_theta_topic", "/tracer/selected_theta")
        self.declare_parameter("mpc_reference_topic", "/tracer/mpc_reference")
        self.declare_parameter("selection_info_topic", "/tracer/runtime_theta_selection_info")

        self.declare_parameter("hold_vx", 0.0)
        self.declare_parameter("hold_yaw_rate", 0.0)
        self.declare_parameter("hold_body_height", 0.300)
        self.declare_parameter("hold_clearance", 0.035)
        self.declare_parameter("hold_enable", 1.0)
        self.declare_parameter("policy_ramp_sec", 1.0)

        self.selector_path = Path(str(self.get_parameter("selector_path").value))
        self.selector = json.loads(self.selector_path.read_text())

        self.runtime_phase = "policy"
        self.prev_runtime_phase = "policy"
        self.phase_switch_time = self.get_clock().now()
        self.requested_profile = str(self.get_parameter("profile").value)
        self.counter = 0.0

        self.selected = self._select(self.requested_profile, reason="initial")

        self.pub_theta = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("selected_theta_topic").value),
            10,
        )
        self.pub_ref = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("mpc_reference_topic").value),
            10,
        )
        self.pub_info = self.create_publisher(
            String,
            str(self.get_parameter("selection_info_topic").value),
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

        self.get_logger().info(
            "[TRACER] runtime-validated policy node started "
            f"profile={self.requested_profile} selector={self.selector_path}"
        )
        self._log_selection()

    def _select(self, requested_profile: str, reason: str):
        profiles = self.selector.get("profiles", {})
        global_default = self.selector["global_default"]

        candidate = profiles.get(requested_profile)
        if candidate is None:
            selected = global_default
            selection_reason = f"unknown_profile:{requested_profile}->global_default"
        elif bool(candidate.get("runtime_allowed", False)):
            selected = candidate
            selection_reason = f"runtime_allowed:{requested_profile}"
        else:
            selected = global_default
            selection_reason = (
                f"runtime_blocked:{requested_profile}"
                f"(feasible={candidate.get('feasible_frac')})"
                f"->global_default:{global_default['profile']}"
            )

        out = {
            "requested_profile": requested_profile,
            "request_reason": reason,
            "selection_reason": selection_reason,
            "selected_profile": selected["profile"],
            "selected_preset": selected["preset"],
            "theta_norm": selected["theta_norm"],
            "feasible_frac": selected["feasible_frac"],
            "runtime_score": selected["runtime_score"],
            "zmin_mean": selected["zmin_mean"],
            "distance_mean": selected["distance_mean"],
        }
        return out

    def _log_selection(self):
        self.get_logger().info(
            "[TRACER] runtime selector "
            f"requested={self.selected['requested_profile']} "
            f"selected={self.selected['selected_profile']} "
            f"reason={self.selected['selection_reason']} "
            f"theta={self.selected['theta_norm']}"
        )

    def _on_beta(self, msg: Float64MultiArray):
        beta = [float(x) for x in msg.data[:3]]
        if len(beta) < 3 or any(not math.isfinite(x) for x in beta):
            self.get_logger().warn(f"[TRACER] ignoring invalid beta={beta}")
            return

        profile, dist = infer_profile_from_beta(beta)
        self.requested_profile = profile
        self.selected = self._select(profile, reason=f"nearest_beta_profile dist={dist:.6f}")
        self._log_selection()

    def _on_runtime_phase(self, msg: String):
        phase = str(msg.data).strip().lower()
        if phase not in {"hold", "stand", "reset", "policy"}:
            self.get_logger().warn(f"[TRACER] ignoring unknown runtime_phase={phase}")
            return

        if phase in {"stand", "reset"}:
            phase = "hold"

        if phase != self.runtime_phase:
            self.prev_runtime_phase = self.runtime_phase
            self.runtime_phase = phase
            self.phase_switch_time = self.get_clock().now()
            self.get_logger().info(
                f"[TRACER] runtime_phase {self.prev_runtime_phase} -> {self.runtime_phase}"
            )

    def _on_timer(self):
        theta = [float(x) for x in self.selected["theta_norm"]]

        hold_ref = [
            float(self.counter),
            float(self.get_parameter("hold_vx").value),
            float(self.get_parameter("hold_yaw_rate").value),
            float(self.get_parameter("hold_body_height").value),
            float(self.get_parameter("hold_clearance").value),
            float(self.get_parameter("hold_enable").value),
        ]
        policy_ref = theta_to_mpc_ref(theta, counter=self.counter)

        if self.runtime_phase == "hold":
            ref = hold_ref
        else:
            ramp_sec = max(0.0, float(self.get_parameter("policy_ramp_sec").value))
            elapsed = (self.get_clock().now() - self.phase_switch_time).nanoseconds * 1.0e-9

            if self.prev_runtime_phase == "hold" and ramp_sec > 1.0e-6 and elapsed < ramp_sec:
                alpha = max(0.0, min(1.0, elapsed / ramp_sec))
                ref = [
                    float(self.counter),
                    hold_ref[1] + alpha * (policy_ref[1] - hold_ref[1]),
                    hold_ref[2] + alpha * (policy_ref[2] - hold_ref[2]),
                    hold_ref[3] + alpha * (policy_ref[3] - hold_ref[3]),
                    hold_ref[4] + alpha * (policy_ref[4] - hold_ref[4]),
                    hold_ref[5] + alpha * (policy_ref[5] - hold_ref[5]),
                ]
            else:
                ref = policy_ref

        theta_msg = Float64MultiArray()
        theta_msg.data = theta
        self.pub_theta.publish(theta_msg)

        ref_msg = Float64MultiArray()
        ref_msg.data = ref
        self.pub_ref.publish(ref_msg)

        info_msg = String()
        info_msg.data = json.dumps(self.selected)
        self.pub_info.publish(info_msg)

        self.counter += 1.0


def main():
    rclpy.init()
    node = RuntimeValidatedPolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
