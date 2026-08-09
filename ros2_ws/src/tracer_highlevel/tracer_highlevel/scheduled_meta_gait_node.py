#!/usr/bin/env python3

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class ScheduledMetaGaitNode(Node):
    """
    Minimal time-scheduled TRACER high-level publisher.

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

    M6.2 v0 intentionally schedules only vx.
    Structural gait parameters remain fixed.
    """

    def __init__(self):
        super().__init__("tracer_scheduled_meta_gait_node")

        self.declare_parameter(
            "command_topic",
            "/tracer/meta_gait_cmd",
        )
        self.declare_parameter(
            "publish_hz",
            10.0,
        )

        self.declare_parameter("segment_1_s", 2.0)
        self.declare_parameter("segment_2_s", 4.0)
        self.declare_parameter("segment_3_s", 6.0)

        self.declare_parameter("vx_0", 0.12)
        self.declare_parameter("vx_1", 0.16)
        self.declare_parameter("vx_2", 0.08)
        self.declare_parameter("vx_3", 0.12)

        self.declare_parameter("yaw_rate", 0.0)
        self.declare_parameter("body_height", 0.30)
        self.declare_parameter("swing_clearance", 0.06)
        self.declare_parameter("gait_period", 1.0 / 1.4)
        self.declare_parameter("duty_factor", 0.65)

        self.command_topic = str(
            self.get_parameter("command_topic").value
        )
        self.publish_hz = float(
            self.get_parameter("publish_hz").value
        )

        self.segment_1_s = float(
            self.get_parameter("segment_1_s").value
        )
        self.segment_2_s = float(
            self.get_parameter("segment_2_s").value
        )
        self.segment_3_s = float(
            self.get_parameter("segment_3_s").value
        )

        if (
            not math.isfinite(self.publish_hz)
            or self.publish_hz <= 0.0
        ):
            raise ValueError(
                "publish_hz must be finite and > 0"
            )

        if not (
            0.0
            < self.segment_1_s
            < self.segment_2_s
            < self.segment_3_s
        ):
            raise ValueError(
                "segment times must satisfy "
                "0 < segment_1_s < segment_2_s "
                "< segment_3_s"
            )

        self.publisher = self.create_publisher(
            Float64MultiArray,
            self.command_topic,
            10,
        )

        self.start_monotonic = time.monotonic()
        self.last_segment = None

        self.timer = self.create_timer(
            1.0 / self.publish_hz,
            self.publish_command,
        )

        self.get_logger().info(
            "TRACER scheduled high-level node started: "
            f"{self.command_topic} @ "
            f"{self.publish_hz:.1f} Hz"
        )

    def p(self, name: str) -> float:
        value = float(
            self.get_parameter(name).value
        )

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        return value

    def elapsed_s(self) -> float:
        return (
            time.monotonic()
            - self.start_monotonic
        )

    def current_segment(
        self,
        elapsed: float,
    ) -> tuple[int, float]:
        if elapsed < self.segment_1_s:
            return 0, self.p("vx_0")

        if elapsed < self.segment_2_s:
            return 1, self.p("vx_1")

        if elapsed < self.segment_3_s:
            return 2, self.p("vx_2")

        return 3, self.p("vx_3")

    def command_values(
        self,
        vx: float,
    ) -> list[float]:
        values = [
            vx,
            self.p("yaw_rate"),
            self.p("body_height"),
            self.p("swing_clearance"),
            self.p("gait_period"),
            self.p("duty_factor"),
        ]

        if values[4] <= 0.0:
            raise ValueError(
                "gait_period must be > 0"
            )

        return values

    def publish_command(self):
        elapsed = self.elapsed_s()
        segment, vx = self.current_segment(
            elapsed
        )

        if segment != self.last_segment:
            self.last_segment = segment

            self.get_logger().info(
                "M6.2 schedule transition: "
                f"segment={segment} "
                f"t={elapsed:.3f}s "
                f"vx={vx:.3f}"
            )

        msg = Float64MultiArray()
        msg.data = self.command_values(vx)

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = ScheduledMetaGaitNode()

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
