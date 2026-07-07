#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


@dataclass(frozen=True)
class Segment:
    name: str
    x_min: float
    x_max: float
    onehot: list[float]
    slope_rad: float
    roughness_proxy: float
    risk_proxy: float


SEGMENTS = [
    Segment("flat",       -1.0e9, 2.0,    [1, 0, 0, 0, 0],  0.0,       0.00, 0.05),
    Segment("upslope",    2.0,    4.0,    [0, 1, 0, 0, 0],  0.087266,  0.05, 0.15),
    Segment("rough",      4.0,    6.0,    [0, 0, 1, 0, 0],  0.0,       0.55, 0.30),
    Segment("downslope",  6.0,    8.0,    [0, 0, 0, 1, 0], -0.087266,  0.05, 0.25),
    Segment("goal_flat",  8.0,    1.0e9,  [0, 0, 0, 0, 1],  0.0,       0.00, 0.05),
]


SEGMENT_ID = {seg.name: i for i, seg in enumerate(SEGMENTS)}
GOAL_SEGMENT = SEGMENTS[-1]


class MixedTerrainContextProvider(Node):
    def __init__(self):
        super().__init__("tracer_mixed_terrain_context_provider_v0")

        self.seq = 0
        self.last_label: str | None = None
        self.goal_flat_latched = False

        self.sub = self.create_subscription(
            Float64MultiArray,
            "/tracer/robot_odom_flat",
            self.on_odom,
            20,
        )

        self.pub_flat = self.create_publisher(
            Float64MultiArray,
            "/tracer/terrain_context_flat",
            10,
        )
        self.pub_label = self.create_publisher(
            String,
            "/tracer/terrain_context_label",
            10,
        )

        self.get_logger().info("mixed terrain context provider v0 started")
        self.get_logger().info(
            "publishing /tracer/terrain_context_flat and /tracer/terrain_context_label"
        )
        self.get_logger().info(
            "context flat layout: "
            "[seq, stamp_sec, x_rel, y_rel, segment_id, "
            "onehot_flat, onehot_upslope, onehot_rough, onehot_downslope, onehot_goal_flat, "
            "slope_rad, roughness_proxy, risk_proxy]"
        )

    def lookup_segment(self, x: float) -> Segment:
        # Once the robot enters the goal zone, latch goal_flat to avoid x≈8.0 boundary jitter.
        if x >= GOAL_SEGMENT.x_min:
            self.goal_flat_latched = True

        if self.goal_flat_latched:
            return GOAL_SEGMENT

        for seg in SEGMENTS:
            if seg.x_min <= x < seg.x_max:
                return seg

        return SEGMENTS[0]

    def on_odom(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 2:
            return

        x = float(msg.data[0])
        y = float(msg.data[1])
        seg = self.lookup_segment(x)

        if seg.name != self.last_label:
            self.get_logger().info(
                f"context switch: x={x:.3f}, y={y:.3f}, segment={seg.name}"
            )
            self.last_label = seg.name

        stamp_sec = self.get_clock().now().nanoseconds * 1.0e-9
        segment_id = SEGMENT_ID[seg.name]

        out = Float64MultiArray()
        out.data = [
            float(self.seq),
            float(stamp_sec),
            float(x),
            float(y),
            float(segment_id),
            *[float(v) for v in seg.onehot],
            float(seg.slope_rad),
            float(seg.roughness_proxy),
            float(seg.risk_proxy),
        ]
        self.pub_flat.publish(out)

        label_msg = String()
        label_msg.data = seg.name
        self.pub_label.publish(label_msg)

        self.seq += 1


def main() -> None:
    rclpy.init()
    node = MixedTerrainContextProvider()
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
