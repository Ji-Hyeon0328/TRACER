#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


def finite_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


class TracerPhaseAOverridePolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("tracer_phase_a_override_policy_node_v0")

        self.declare_parameter("terrain", "unknown")
        self.declare_parameter("publish_hz", 20.0)

        self.declare_parameter("vx", 0.0)
        self.declare_parameter("yaw_rate", 0.0)
        self.declare_parameter("body_height", 0.30)
        self.declare_parameter("swing_clearance", 0.03)
        self.declare_parameter("enable", 1.0)

        self.declare_parameter("mpc_topic", "/tracer/mpc_reference")
        self.declare_parameter("status_topic", "/tracer/phase_a_policy_status")
        self.declare_parameter("debug_topic", "/tracer/highlevel_debug")

        self.terrain = str(self.get_parameter("terrain").value)
        self.publish_hz = finite_float(self.get_parameter("publish_hz").value, 20.0)

        self.vx = finite_float(self.get_parameter("vx").value, 0.0)
        self.yaw_rate = finite_float(self.get_parameter("yaw_rate").value, 0.0)
        self.body_height = finite_float(self.get_parameter("body_height").value, 0.30)
        self.swing_clearance = finite_float(self.get_parameter("swing_clearance").value, 0.03)
        self.enable = finite_float(self.get_parameter("enable").value, 1.0)

        self.mpc_topic = str(self.get_parameter("mpc_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)

        self.pub = self.create_publisher(Float64MultiArray, self.mpc_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        self.counter = 0.0
        self.t0 = time.time()

        period = 1.0 / max(1.0, self.publish_hz)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            "Phase-A override policy node started. "
            f"terrain={self.terrain} vx={self.vx:.3f} yaw={self.yaw_rate:.3f} "
            f"h={self.body_height:.3f} clr={self.swing_clearance:.3f} enable={self.enable:.1f} "
            f"mpc_topic={self.mpc_topic} status_topic={self.status_topic} debug_topic={self.debug_topic} "
            f"hz={self.publish_hz:.1f}"
        )

    def on_timer(self) -> None:
        self.counter += 1.0

        arr = [
            self.counter,
            self.vx,
            self.yaw_rate,
            self.body_height,
            self.swing_clearance,
            self.enable,
        ]

        self.pub.publish(Float64MultiArray(data=arr))

        status = {
            "final_guard_pass": True,
            "mode": "manual_override_action",
            "selection_status": "manual_override_for_runtime_probe",
            "terrain": self.terrain,
            "policy_node": "tracer_phase_a_override_policy_node_v0",
            "wall_time": time.time(),
            "elapsed_sec": time.time() - self.t0,
            "mpc_reference": {
                "counter": self.counter,
                "vx": self.vx,
                "yaw_rate": self.yaw_rate,
                "body_height": self.body_height,
                "swing_clearance": self.swing_clearance,
                "enable": self.enable,
            },
            "note": (
                "Manual override bypasses RAM-aware stack selection. "
                "Use for runtime candidate probing; export validated candidates back to stack later."
            ),
        }

        msg = String()
        msg.data = json.dumps(status, sort_keys=True)
        self.status_pub.publish(msg)
        self.debug_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = TracerPhaseAOverridePolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
