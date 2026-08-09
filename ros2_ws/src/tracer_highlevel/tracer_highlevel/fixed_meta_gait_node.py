#!/usr/bin/env python3

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class FixedMetaGaitNode(Node):
    """
    Minimal TRACER high-level publisher.

    Output:
      /tracer/meta_gait_cmd

    Layout:
      [
        vx,
        yaw_rate,
        body_height,
        swing_clearance,
        gait_period,
        duty_factor,
      ]

    This node intentionally knows nothing about PyMPC,
    M4, TransitionManager, or torque control.
    """

    def __init__(self):
        super().__init__(
            "tracer_fixed_meta_gait_node"
        )

        self.declare_parameter(
            "command_topic",
            "/tracer/meta_gait_cmd",
        )

        self.declare_parameter(
            "publish_hz",
            10.0,
        )

        self.declare_parameter(
            "vx",
            0.12,
        )

        self.declare_parameter(
            "yaw_rate",
            0.0,
        )

        self.declare_parameter(
            "body_height",
            0.30,
        )

        self.declare_parameter(
            "swing_clearance",
            0.06,
        )

        self.declare_parameter(
            "gait_period",
            1.0 / 1.4,
        )

        self.declare_parameter(
            "duty_factor",
            0.65,
        )

        self.command_topic = str(
            self.get_parameter(
                "command_topic"
            ).value
        )

        self.publish_hz = float(
            self.get_parameter(
                "publish_hz"
            ).value
        )

        if (
            not math.isfinite(
                self.publish_hz
            )
            or self.publish_hz <= 0.0
        ):
            raise ValueError(
                "publish_hz must be finite and > 0"
            )

        self.publisher = self.create_publisher(
            Float64MultiArray,
            self.command_topic,
            10,
        )

        self.timer = self.create_timer(
            1.0 / self.publish_hz,
            self.publish_command,
        )

        values = self.command_values()

        self.get_logger().info(
            "TRACER fixed high-level node started: "
            f"{self.command_topic} @ "
            f"{self.publish_hz:.1f} Hz; "
            f"vx={values[0]:.3f}, "
            f"yaw={values[1]:.3f}, "
            f"h={values[2]:.3f}, "
            f"clr={values[3]:.3f}, "
            f"T={values[4]:.6f}, "
            f"D={values[5]:.3f}"
        )

    def command_values(
        self,
    ) -> list[float]:
        names = (
            "vx",
            "yaw_rate",
            "body_height",
            "swing_clearance",
            "gait_period",
            "duty_factor",
        )

        values = [
            float(
                self.get_parameter(
                    name
                ).value
            )
            for name in names
        ]

        for name, value in zip(
            names,
            values,
        ):
            if not math.isfinite(value):
                raise ValueError(
                    f"{name} must be finite"
                )

        if values[4] <= 0.0:
            raise ValueError(
                "gait_period must be > 0"
            )

        return values

    def publish_command(self):
        msg = Float64MultiArray()
        msg.data = self.command_values()

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = FixedMetaGaitNode()

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
