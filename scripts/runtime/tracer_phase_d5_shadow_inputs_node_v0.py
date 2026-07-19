#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray


BETA_TABLE = {
    "flat":       [0.45, 0.25, 0.30],
    "start_flat": [0.45, 0.25, 0.30],
    "rough":      [0.25, 0.55, 0.20],
    "upslope":    [0.35, 0.40, 0.25],
    "downslope":  [0.25, 0.60, 0.15],
    "goal_flat":  [0.20, 0.50, 0.30],
    "unknown":    [0.33, 0.34, 0.33],
}

# [slip_proxy, roughness_proxy, uncertainty_sigma]
RAM_TABLE = {
    "flat":       [0.00, 0.05, 0.05],
    "start_flat": [0.00, 0.05, 0.05],
    "rough":      [0.10, 0.40, 0.35],
    "upslope":    [0.10, 0.20, 0.20],
    "downslope":  [0.25, 0.15, 0.30],
    "goal_flat":  [0.00, 0.05, 0.05],
    "unknown":    [0.10, 0.10, 0.20],
}


class ShadowInputs(Node):
    def __init__(self):
        super().__init__("tracer_phase_d5_shadow_inputs_v0")

        self.context_topic = os.environ.get("TRACER_CONTEXT_LABEL_TOPIC", "/tracer/terrain_context_label")
        self.beta_topic = os.environ.get("TRACER_PHASE_D5_BETA_TOPIC", "/tracer/objective_beta")
        self.ram_topic = os.environ.get("TRACER_PHASE_D5_RAM_TOPIC", "/tracer/ram_mismatch")
        self.pub_hz = float(os.environ.get("TRACER_PHASE_D5_PUB_HZ", "10.0"))

        self.label = os.environ.get("TRACER_PHASE_D5_DEFAULT_CONTEXT", "unknown")

        self.create_subscription(String, self.context_topic, self.on_context, 10)
        self.beta_pub = self.create_publisher(Float64MultiArray, self.beta_topic, 10)
        self.ram_pub = self.create_publisher(Float64MultiArray, self.ram_topic, 10)

        self.timer = self.create_timer(1.0 / self.pub_hz, self.on_timer)

        self.get_logger().info(f"context_topic={self.context_topic}")
        self.get_logger().info(f"beta_topic={self.beta_topic}")
        self.get_logger().info(f"ram_topic={self.ram_topic}")

    def on_context(self, msg):
        label = msg.data.strip()
        self.label = label if label else "unknown"

    def on_timer(self):
        beta = BETA_TABLE.get(self.label, BETA_TABLE["unknown"])
        ram = RAM_TABLE.get(self.label, RAM_TABLE["unknown"])

        beta_msg = Float64MultiArray()
        beta_msg.data = [float(x) for x in beta]
        self.beta_pub.publish(beta_msg)

        ram_msg = Float64MultiArray()
        ram_msg.data = [float(x) for x in ram]
        self.ram_pub.publish(ram_msg)


def main():
    rclpy.init()
    node = ShadowInputs()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
