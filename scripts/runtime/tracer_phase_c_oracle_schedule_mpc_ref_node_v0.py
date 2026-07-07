#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


COMMANDS = {
    "flat":       (0.20, 0.000, 0.320, 0.045),
    "upslope":    (0.16, 0.000, 0.325, 0.055),
    "rough":      (0.12, 0.000, 0.335, 0.075),
    "downslope":  (0.10, 0.000, 0.330, 0.060),
    "goal_flat":  (0.00, 0.000, 0.320, 0.045),
}


class PhaseCOracleSchedule(Node):
    def __init__(self, goal_x: float, pub_hz: float):
        super().__init__("tracer_phase_c_oracle_schedule_mpc_ref_node_v0")

        self.goal_x = float(goal_x)
        self.seq = 0
        self.label = "flat"
        self.x_rel = 0.0
        self.y_rel = 0.0
        self.stopped = False

        self.pub = self.create_publisher(Float64MultiArray, "/tracer/mpc_reference", 10)

        self.sub_label = self.create_subscription(
            String,
            "/tracer/terrain_context_label",
            self.on_label,
            10,
        )

        self.sub_odom = self.create_subscription(
            Float64MultiArray,
            "/tracer/robot_odom_flat",
            self.on_odom,
            20,
        )

        self.timer = self.create_timer(1.0 / float(pub_hz), self.on_timer)

        self.get_logger().info("Phase-C oracle schedule MPC ref node started")
        self.get_logger().info(f"goal_x={self.goal_x:.3f}, pub_hz={pub_hz:.1f}")
        self.get_logger().info("publishing /tracer/mpc_reference")

    def on_label(self, msg: String) -> None:
        old = self.label
        self.label = msg.data.strip()
        if self.label != old:
            self.get_logger().info(f"terrain label switch: {old} -> {self.label}")

    def on_odom(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 2:
            self.x_rel = float(msg.data[0])
            self.y_rel = float(msg.data[1])

    def on_timer(self) -> None:
        label = self.label if self.label in COMMANDS else "flat"

        vx, yaw_rate, body_h, clearance = COMMANDS[label]

        if self.x_rel >= self.goal_x or label == "goal_flat":
            vx = 0.0
            yaw_rate = 0.0
            self.stopped = True

        out = Float64MultiArray()
        out.data = [
            float(self.seq),
            float(vx),
            float(yaw_rate),
            float(body_h),
            float(clearance),
            1.0,
        ]
        self.pub.publish(out)

        if self.seq % 20 == 0:
            self.get_logger().info(
                f"seq={self.seq} x={self.x_rel:.3f} y={self.y_rel:.3f} "
                f"label={label} cmd=[vx={vx:.3f}, yaw={yaw_rate:.3f}, "
                f"h={body_h:.3f}, clr={clearance:.3f}] stopped={self.stopped}"
            )

        self.seq += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-x", type=float, default=8.4)
    parser.add_argument("--pub-hz", type=float, default=10.0)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = PhaseCOracleSchedule(goal_x=args.goal_x, pub_hz=args.pub_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt: publishing stop command before shutdown")
        try:
            stop = Float64MultiArray()
            stop.data = [9999.0, 0.0, 0.0, 0.320, 0.045, 1.0]
            node.pub.publish(stop)
            time.sleep(0.1)
        except Exception as exc:
            node.get_logger().warn(f"failed to publish stop during shutdown: {exc}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
