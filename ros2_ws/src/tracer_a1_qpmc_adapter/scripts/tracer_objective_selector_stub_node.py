#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def normalize_beta(beta):
    beta = [max(0.0, float(x)) for x in beta]
    s = sum(beta)
    if s <= 1e-9:
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    return [x / s for x in beta]


class TracerObjectiveSelectorStubNode(Node):
    """
    Objective Selector stub.

    Output:
      /tracer/objective_weights:
        [beta_motion, beta_stability, beta_energy]

    Inputs:
      /tracer/objective_mode:
        [0] balanced
        [1] motion-prioritized
        [2] stability-prioritized
        [3] energy-prioritized

      /tracer/objective_weights_cmd:
        direct [beta_motion, beta_stability, beta_energy]
    """

    def __init__(self):
        super().__init__("tracer_objective_selector_stub_node")

        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("objective_topic", "/tracer/objective_weights")
        self.declare_parameter("mode_topic", "/tracer/objective_mode")
        self.declare_parameter("weights_cmd_topic", "/tracer/objective_weights_cmd")
        self.declare_parameter("default_mode", 0)

        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.objective_topic = self.get_parameter("objective_topic").value
        self.mode_topic = self.get_parameter("mode_topic").value
        self.weights_cmd_topic = self.get_parameter("weights_cmd_topic").value

        self.beta = self.mode_to_beta(int(self.get_parameter("default_mode").value))

        self.pub = self.create_publisher(Float64MultiArray, self.objective_topic, 10)

        self.create_subscription(
            Float64MultiArray,
            self.mode_topic,
            self.mode_callback,
            10,
        )

        self.create_subscription(
            Float64MultiArray,
            self.weights_cmd_topic,
            self.weights_callback,
            10,
        )

        self.timer = self.create_timer(
            1.0 / max(1.0, self.publish_hz),
            self.timer_callback,
        )

        self.get_logger().info(
            f"objective selector stub -> {self.objective_topic}, initial beta={self.beta}"
        )

    def mode_to_beta(self, mode):
        if mode == 1:
            # Motion-prioritized
            return normalize_beta([0.60, 0.25, 0.15])
        if mode == 2:
            # Stability-prioritized
            return normalize_beta([0.20, 0.65, 0.15])
        if mode == 3:
            # Energy-prioritized
            return normalize_beta([0.20, 0.25, 0.55])

        # Balanced
        return normalize_beta([1.0, 1.0, 1.0])

    def mode_callback(self, msg):
        if len(msg.data) < 1:
            return

        mode = int(msg.data[0])
        self.beta = self.mode_to_beta(mode)
        self.get_logger().info(f"objective mode={mode} -> beta={self.beta}")

    def weights_callback(self, msg):
        if len(msg.data) < 3:
            self.get_logger().warn(
                f"objective_weights_cmd expects [beta_motion,beta_stability,beta_energy], got {len(msg.data)}"
            )
            return

        self.beta = normalize_beta(msg.data[:3])
        self.get_logger().info(f"direct objective weights -> beta={self.beta}")

    def timer_callback(self):
        msg = Float64MultiArray()
        msg.data = list(self.beta)
        self.pub.publish(msg)

        self.get_logger().info(
            f"beta motion={self.beta[0]:.2f} stability={self.beta[1]:.2f} energy={self.beta[2]:.2f}",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = TracerObjectiveSelectorStubNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
