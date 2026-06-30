#!/usr/bin/env python3
from __future__ import annotations

import json

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


class ObjectiveSelectorBetaNode(Node):
    def __init__(self):
        super().__init__("tracer_objective_selector_beta_node_v0")

        self.declare_parameter("profile", "balanced")
        self.declare_parameter("publish_hz", 5.0)
        self.declare_parameter("beta_topic", "/tracer/objective_beta")
        self.declare_parameter("profile_command_topic", "/tracer/objective_profile_command")
        self.declare_parameter("info_topic", "/tracer/objective_selector_info")

        self.profile = str(self.get_parameter("profile").value)
        if self.profile not in PROFILE_BETA:
            self.get_logger().warn(
                f"[TRACER] unknown initial profile={self.profile}; using balanced"
            )
            self.profile = "balanced"

        self.pub_beta = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("beta_topic").value),
            10,
        )
        self.pub_info = self.create_publisher(
            String,
            str(self.get_parameter("info_topic").value),
            10,
        )

        self.create_subscription(
            String,
            str(self.get_parameter("profile_command_topic").value),
            self._on_profile_command,
            10,
        )

        hz = max(0.5, float(self.get_parameter("publish_hz").value))
        self.timer = self.create_timer(1.0 / hz, self._on_timer)

        self.get_logger().info(
            f"[TRACER] objective selector beta node started profile={self.profile}"
        )

    def _on_profile_command(self, msg: String):
        profile = str(msg.data).strip().lower()
        if profile not in PROFILE_BETA:
            self.get_logger().warn(f"[TRACER] ignoring unknown profile command={profile}")
            return

        if profile != self.profile:
            self.profile = profile
            self.get_logger().info(f"[TRACER] objective profile -> {self.profile}")

    def _on_timer(self):
        beta = PROFILE_BETA[self.profile]

        beta_msg = Float64MultiArray()
        beta_msg.data = [float(x) for x in beta]
        self.pub_beta.publish(beta_msg)

        info = {
            "node": "tracer_objective_selector_beta_node_v0",
            "profile": self.profile,
            "beta": beta,
            "note": "Rule/profile-based Objective Selector v0. Replace with learned beta predictor later.",
        }
        info_msg = String()
        info_msg.data = json.dumps(info)
        self.pub_info.publish(info_msg)


def main():
    rclpy.init()
    node = ObjectiveSelectorBetaNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
