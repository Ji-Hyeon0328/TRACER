#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


class ProprioVectorToRamScalarFlat(Node):
    """
    Convert live TRACER proprio_vector into the compact proprio_flat layout used by
    scalar RAM v3 shadow smoke tests.

    Expected /tracer/proprio_vector layout:
      [stamp_wall, base_x, base_y, base_z, rpy(3), lin_vel(3), ang_vel(3), joints...]

    Output /tracer/proprio_flat layout:
      [base_x, base_y, base_z, proprio_abs_mean, proprio_abs_max]
    """

    def __init__(self):
        super().__init__("tracer_proprio_vector_to_ram_scalar_flat")

        self.declare_parameter(
            "input_topic",
            os.environ.get("TRACER_PROPRIO_VECTOR_TOPIC", "/tracer/proprio_vector"),
        )
        self.declare_parameter(
            "output_topic",
            os.environ.get("TRACER_RAM_SCALAR_V3_PROPRIO_FLAT_TOPIC", "/tracer/proprio_flat"),
        )

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)

        self.pub = self.create_publisher(Float64MultiArray, self.output_topic, 10)
        self.create_subscription(Float64MultiArray, self.input_topic, self.on_msg, 20)

        self.get_logger().info(
            f"proprio vector -> RAM scalar flat converter started "
            f"input={self.input_topic} output={self.output_topic}"
        )

    def on_msg(self, msg: Float64MultiArray):
        xs = list(msg.data)

        # Default safe values.
        base_x = 0.0
        base_y = 0.0
        base_z = 0.0
        vals_for_abs = []

        if len(xs) >= 4:
            # Live qwerty layout includes stamp_wall at index 0.
            base_x = f(xs[1])
            base_y = f(xs[2])
            base_z = f(xs[3])
            vals_for_abs = [abs(f(v)) for v in xs[1:]]
        elif len(xs) >= 3:
            # Fallback for already-flat-ish vectors without stamp.
            base_x = f(xs[0])
            base_y = f(xs[1])
            base_z = f(xs[2])
            vals_for_abs = [abs(f(v)) for v in xs]

        proprio_abs_mean = mean(vals_for_abs) if vals_for_abs else 0.0
        proprio_abs_max = max(vals_for_abs) if vals_for_abs else 0.0

        out = Float64MultiArray()
        out.data = [
            base_x,
            base_y,
            base_z,
            proprio_abs_mean,
            proprio_abs_max,
        ]
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ProprioVectorToRamScalarFlat()
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
