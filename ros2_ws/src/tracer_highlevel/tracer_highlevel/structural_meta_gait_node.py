#!/usr/bin/env python3

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class StructuralMetaGaitNode(Node):
    """M6.2 structural high-level transition test producer."""

    def __init__(self):
        super().__init__("tracer_structural_meta_gait_node")

        self.publisher = self.create_publisher(
            Float64MultiArray,
            "/tracer/meta_gait_cmd",
            10,
        )

        self.t0 = time.monotonic()
        self.last_segment = None

        self.timer = self.create_timer(
            0.1,
            self.publish_command,
        )

        self.get_logger().info(
            "M6.2 structural HL started: "
            "f=1.40,D=0.65 -> f=1.10,D=0.60 at t=3.0s"
        )

    def publish_command(self):
        elapsed = time.monotonic() - self.t0

        if elapsed < 3.0:
            segment = 0
            gait_period = 1.0 / 1.4
            duty_factor = 0.65
        else:
            segment = 1
            gait_period = 1.0 / 1.1
            duty_factor = 0.60

        if segment != self.last_segment:
            self.last_segment = segment

            self.get_logger().info(
                "M6.2 structural request: "
                f"segment={segment} "
                f"t={elapsed:.3f}s "
                f"f={1.0 / gait_period:.3f} "
                f"D={duty_factor:.3f}"
            )

        msg = Float64MultiArray()
        msg.data = [
            0.20,          # vx
            0.0,           # yaw_rate
            0.30,          # body_height
            0.06,          # swing_clearance
            gait_period,
            duty_factor,
        ]

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StructuralMetaGaitNode()

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
