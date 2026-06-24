#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


ROOT = Path(os.environ.get("TRACER_ROOT", str(Path.home() / "Tracer/TRACER")))

POLICY_JSON = Path(
    os.environ.get(
        "TRACER_GATE_POLICY_JSON",
        str(ROOT / "configs/highlevel_policy/tracer_fusion_policy_v1.json"),
    )
).expanduser()

POLICY_TERRAIN = os.environ.get("TRACER_GATE_POLICY_TERRAIN", "flat_normal")

# RAM risk message layout:
# [
#   future_risk, future_slip, future_invalid, run_valid,
#   run_fallen, recovery_needed, sigma_mean, rho_norm,
#   style_score_pred, control_risk_raw, control_risk_ema
# ]

MODE_CODE = {
    "unknown": -1.0,
    "locomotion": 1.0,
    "cautious_locomotion": 2.0,
    "recovery_needed": 3.0,
    "avoid_required": 4.0,
}

SEMANTIC_CODE = {
    "unknown": -1.0,
    "validated_locomotion": 1.0,
    "no_valid_simple_primitive": 2.0,
    "cautious_locomotion": 3.0,
    "recovery_needed": 4.0,
    "candidate_conditional_micro_brake": 5.0,
    "failed_candidate_conditional_micro_brake": 6.0,
    "no_valid_high_level_velocity_primitive": 7.0,
}

GATE_LEVEL_CODE = {
    "stable": 0.0,
    "caution": 1.0,
    "unstable": 2.0,
}

ACTION_CODE = {
    "keep": 0.0,
    "would_cautious": 1.0,
    "would_conservative_probe": 2.0,
    "prior_avoid_keep": 3.0,
    "prior_recovery_keep": 4.0,
    "no_valid_keep": 5.0,
    "failed_candidate_keep": 6.0,
    "unknown": -1.0,
}


class RamGateMonitor(Node):
    def __init__(self) -> None:
        super().__init__("tracer_ram_gate_monitor_node")

        self.policy_terrain = POLICY_TERRAIN
        self.policy_json = POLICY_JSON
        self.policy_entry = self._load_policy_entry()

        self.caution_ctrl = float(os.environ.get("TRACER_GATE_CAUTION_CTRL", "0.05"))
        self.unstable_ctrl = float(os.environ.get("TRACER_GATE_UNSTABLE_CTRL", "0.25"))

        self.caution_fallen = float(os.environ.get("TRACER_GATE_CAUTION_FALLEN", "0.30"))
        self.unstable_fallen = float(os.environ.get("TRACER_GATE_UNSTABLE_FALLEN", "0.70"))

        self.caution_sigma = float(os.environ.get("TRACER_GATE_CAUTION_SIGMA", "0.50"))
        self.unstable_sigma = float(os.environ.get("TRACER_GATE_UNSTABLE_SIGMA", "0.85"))

        self.caution_recovery = float(os.environ.get("TRACER_GATE_CAUTION_RECOVERY", "0.20"))
        self.unstable_recovery = float(os.environ.get("TRACER_GATE_UNSTABLE_RECOVERY", "0.60"))

        self.pub_advice = self.create_publisher(
            Float64MultiArray,
            "/tracer/ram_gate_advice",
            10,
        )
        self.pub_text = self.create_publisher(
            String,
            "/tracer/ram_gate_advice_text",
            10,
        )

        self.sub_ram = self.create_subscription(
            Float64MultiArray,
            "/tracer/ram_risk",
            self.on_ram_risk,
            10,
        )

        self.last_log_time = 0.0
        self.msg_count = 0
        self.skip_initial_msgs = int(os.environ.get("TRACER_GATE_SKIP_INITIAL_MSGS", "3"))

        self.get_logger().info(
            "RAM gate monitor started: "
            f"terrain={self.policy_terrain}, "
            f"policy={self.policy_json}, "
            f"fused_mode={self.policy_entry.get('fused_mode', 'unknown')}, "
            f"semantic={self.policy_entry.get('semantic_mode', 'unknown')}"
        )
        self.get_logger().info(
            "thresholds: "
            f"ctrl caution/unstable={self.caution_ctrl}/{self.unstable_ctrl}, "
            f"fallen caution/unstable={self.caution_fallen}/{self.unstable_fallen}, "
            f"sigma caution/unstable={self.caution_sigma}/{self.unstable_sigma}, "
            f"recovery caution/unstable={self.caution_recovery}/{self.unstable_recovery}"
        )

    def _load_policy_entry(self) -> Dict:
        if not self.policy_json.exists():
            self.get_logger().warn(f"policy json not found: {self.policy_json}")
            return {}

        try:
            data = json.loads(self.policy_json.read_text())
            return data.get("terrains", {}).get(self.policy_terrain, {})
        except Exception as e:
            self.get_logger().warn(f"failed to load policy json: {e}")
            return {}

    def classify_ram(
        self,
        ctrl_ema: float,
        fallen: float,
        recovery: float,
        sigma: float,
    ) -> str:
        if (
            ctrl_ema >= self.unstable_ctrl
            or fallen >= self.unstable_fallen
            or recovery >= self.unstable_recovery
            or sigma >= self.unstable_sigma
        ):
            return "unstable"

        if (
            ctrl_ema >= self.caution_ctrl
            or fallen >= self.caution_fallen
            or recovery >= self.caution_recovery
            or sigma >= self.caution_sigma
        ):
            return "caution"

        return "stable"

    def decide_action(self, ram_level: str) -> Tuple[str, float, float, float, float]:
        fused_mode = self.policy_entry.get("fused_mode", "unknown")
        semantic_mode = self.policy_entry.get("semantic_mode", "unknown")

        # Monitor-only node:
        # this only recommends what would have happened.
        # It never publishes /tracer/mpc_reference.

        no_valid_semantics = {
            "no_valid_simple_primitive",
            "no_valid_high_level_velocity_primitive",
        }

        failed_candidate_semantics = {
            "failed_candidate_conditional_micro_brake",
        }

        if fused_mode == "avoid_required" or semantic_mode in no_valid_semantics:
            return "no_valid_keep", 0.0, 1.0, 0.0, 0.0

        if semantic_mode in failed_candidate_semantics:
            return "failed_candidate_keep", 0.0, 1.0, 0.0, 0.0

        if fused_mode == "recovery_needed" or semantic_mode == "recovery_needed":
            return "prior_recovery_keep", 0.0, 1.0, 0.0, 0.0

        if ram_level == "unstable":
            # Do not jump directly to avoid. First recommend conservative probing.
            return "would_conservative_probe", 1.0, 0.35, 0.015, 0.030

        if ram_level == "caution":
            return "would_cautious", 1.0, 0.60, 0.010, 0.020

        return "keep", 0.0, 1.0, 0.0, 0.0

    def on_ram_risk(self, msg: Float64MultiArray) -> None:
        self.msg_count += 1
        if self.msg_count <= self.skip_initial_msgs:
            if self.msg_count == 1:
                self.get_logger().info(
                    f"skipping initial RAM messages: skip_initial_msgs={self.skip_initial_msgs}"
                )
            return

        data = list(msg.data)
        if len(data) < 9:
            self.get_logger().warn(f"/tracer/ram_risk too short: len={len(data)}")
            return

        future_risk = float(data[0])
        future_slip = float(data[1])
        future_invalid = float(data[2])
        run_valid = float(data[3])
        fallen = float(data[4])
        recovery = float(data[5])
        sigma = float(data[6])
        rho_norm = float(data[7])
        style_score = float(data[8])

        if len(data) >= 11:
            ctrl_risk = float(data[9])
            ctrl_ema = float(data[10])
        else:
            ctrl_risk = 0.50 * future_risk + 0.30 * fallen + 0.20 * recovery
            ctrl_ema = ctrl_risk

        fused_mode = self.policy_entry.get("fused_mode", "unknown")
        semantic_mode = self.policy_entry.get("semantic_mode", "unknown")
        suggested_style = self.policy_entry.get("suggested_style", "unknown")

        ram_level = self.classify_ram(
            ctrl_ema=ctrl_ema,
            fallen=fallen,
            recovery=recovery,
            sigma=sigma,
        )
        action, would_override, vx_scale, body_h_delta, clearance_delta = self.decide_action(ram_level)

        advice = Float64MultiArray()
        advice.data = [
            MODE_CODE.get(fused_mode, MODE_CODE["unknown"]),
            SEMANTIC_CODE.get(semantic_mode, SEMANTIC_CODE["unknown"]),
            GATE_LEVEL_CODE[ram_level],
            ACTION_CODE[action],
            float(would_override),
            float(ctrl_risk),
            float(ctrl_ema),
            float(future_risk),
            float(fallen),
            float(recovery),
            float(sigma),
            float(rho_norm),
            float(style_score),
            float(vx_scale),
            float(body_h_delta),
            float(clearance_delta),
            float(future_slip),
            float(future_invalid),
            float(run_valid),
        ]
        self.pub_advice.publish(advice)

        text = (
            f"terrain={self.policy_terrain} "
            f"v1_mode={fused_mode} "
            f"semantic={semantic_mode} "
            f"style={suggested_style} "
            f"ram_level={ram_level} "
            f"action={action} "
            f"would_override={would_override:.0f} "
            f"ctrl_ema={ctrl_ema:.3f} "
            f"ctrl_risk={ctrl_risk:.3f} "
            f"fallen={fallen:.3f} "
            f"recovery={recovery:.3f} "
            f"sigma={sigma:.3f} "
            f"vx_scale={vx_scale:.2f} "
            f"h_delta={body_h_delta:.3f} "
            f"clr_delta={clearance_delta:.3f}"
        )

        text_msg = String()
        text_msg.data = text
        self.pub_text.publish(text_msg)

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_log_time > 1.0:
            self.get_logger().info(text)
            self.last_log_time = now


def main() -> None:
    rclpy.init()
    node = RamGateMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
