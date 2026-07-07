#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class PhaseCFixedMpcRef(Node):
    def __init__(
        self,
        vx: float,
        yaw_rate: float,
        body_height: float,
        clearance: float,
        goal_x: float,
        pub_hz: float,
    ):
        super().__init__("tracer_phase_c_fixed_mpc_ref_node_v0")

        self.vx = float(vx)
        self.yaw_rate = float(yaw_rate)
        self.body_height = float(body_height)
        self.clearance = float(clearance)
        self.goal_x = float(goal_x)

        self.seq = 0
        self.x_rel = 0.0
        self.y_rel = 0.0
        self.stopped = False

        self.pub = self.create_publisher(Float64MultiArray, "/tracer/mpc_reference", 10)

        self.sub_odom = self.create_subscription(
            Float64MultiArray,
            "/tracer/robot_odom_flat",
            self.on_odom,
            20,
        )

        self.timer = self.create_timer(1.0 / float(pub_hz), self.on_timer)

        self.get_logger().info("Phase-C fixed MPC ref baseline node started")
        self.get_logger().info(
            f"cmd vx={self.vx:.3f}, yaw={self.yaw_rate:.3f}, "
            f"h={self.body_height:.3f}, clr={self.clearance:.3f}, "
            f"goal_x={self.goal_x:.3f}, pub_hz={pub_hz:.1f}"
        )

    def on_odom(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 2:
            self.x_rel = float(msg.data[0])
            self.y_rel = float(msg.data[1])

    def on_timer(self) -> None:
        vx = self.vx
        yaw_rate = self.yaw_rate

        if self.x_rel >= self.goal_x:
            vx = 0.0
            yaw_rate = 0.0
            self.stopped = True

        out = Float64MultiArray()
        out.data = [
            float(self.seq),
            float(vx),
            float(yaw_rate),
            float(self.body_height),
            float(self.clearance),
            1.0,
        ]
        self.pub.publish(out)

        if self.seq % 20 == 0:
            self.get_logger().info(
                f"seq={self.seq} x={self.x_rel:.3f} y={self.y_rel:.3f} "
                f"cmd=[vx={vx:.3f}, yaw={yaw_rate:.3f}, "
                f"h={self.body_height:.3f}, clr={self.clearance:.3f}] "
                f"stopped={self.stopped}"
            )

        self.seq += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vx", type=float, default=0.16)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--body-height", type=float, default=0.32)
    parser.add_argument("--clearance", type=float, default=0.045)
    parser.add_argument("--goal-x", type=float, default=8.4)
    parser.add_argument("--pub-hz", type=float, default=10.0)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = PhaseCFixedMpcRef(
        vx=args.vx,
        yaw_rate=args.yaw_rate,
        body_height=args.body_height,
        clearance=args.clearance,
        goal_x=args.goal_x,
        pub_hz=args.pub_hz,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt: publishing stop command before shutdown")
        try:
            stop = Float64MultiArray()
            stop.data = [9999.0, 0.0, 0.0, args.body_height, args.clearance, 1.0]
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
