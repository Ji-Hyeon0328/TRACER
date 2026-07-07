#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float64MultiArray, String


# beta = [velocity, stability, energy]
BETA_TABLE = {
    "flat":       (0.60, 0.25, 0.15),
    "upslope":    (0.42, 0.43, 0.15),
    "rough":      (0.25, 0.62, 0.13),
    "downslope":  (0.22, 0.65, 0.13),
    "goal_flat":  (0.05, 0.80, 0.15),
}


class PhaseCBetaSchedule(Node):
    def __init__(self, goal_x: float, pub_hz: float):
        super().__init__("tracer_phase_c_beta_schedule_mpc_ref_node_v0")

        self.goal_x = float(goal_x)
        self.seq = 0
        self.label = "flat"
        self.x_rel = 0.0
        self.y_rel = 0.0
        self.stopped = False

        self.pub = self.create_publisher(Float64MultiArray, "/tracer/mpc_reference", 10)
        self.beta_pub = self.create_publisher(Float64MultiArray, "/tracer/objective_weights", 10)

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

        self.get_logger().info("Phase-C beta-aware schedule MPC ref node started")
        self.get_logger().info(f"goal_x={self.goal_x:.3f}, pub_hz={pub_hz:.1f}")

    def on_label(self, msg: String) -> None:
        old = self.label
        self.label = msg.data.strip()
        if self.label != old:
            self.get_logger().info(f"terrain label switch: {old} -> {self.label}")

    def on_odom(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 2:
            self.x_rel = float(msg.data[0])
            self.y_rel = float(msg.data[1])

    @staticmethod
    def command_from_beta(label: str, beta: tuple[float, float, float]):
        beta_v, beta_s, beta_e = beta

        # Smooth beta-to-command scaffold.
        vx = 0.06 + 0.24 * beta_v - 0.04 * beta_s
        vx = max(0.06, min(0.20, vx))

        body_h = 0.315 + 0.030 * beta_s
        clearance = 0.040 + 0.055 * beta_s

        # Terrain guards.
        if label == "upslope":
            vx = min(vx, 0.16)
            clearance = max(clearance, 0.055)
        elif label == "rough":
            vx = min(vx, 0.12)
            body_h = max(body_h, 0.333)
            clearance = max(clearance, 0.073)
        elif label == "downslope":
            vx = min(vx, 0.11)
            body_h = max(body_h, 0.330)
            clearance = max(clearance, 0.060)
        elif label == "goal_flat":
            vx = 0.0

        return vx, 0.0, body_h, clearance

    def on_timer(self) -> None:
        label = self.label if self.label in BETA_TABLE else "flat"
        beta = BETA_TABLE[label]

        vx, yaw_rate, body_h, clearance = self.command_from_beta(label, beta)

        if self.x_rel >= self.goal_x or label == "goal_flat":
            vx = 0.0
            yaw_rate = 0.0
            self.stopped = True

        beta_msg = Float64MultiArray()
        beta_msg.data = [float(self.seq), float(beta[0]), float(beta[1]), float(beta[2])]
        self.beta_pub.publish(beta_msg)

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
                f"label={label} beta=[{beta[0]:.2f},{beta[1]:.2f},{beta[2]:.2f}] "
                f"cmd=[vx={vx:.3f}, yaw={yaw_rate:.3f}, h={body_h:.3f}, clr={clearance:.3f}] "
                f"stopped={self.stopped}"
            )

        self.seq += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-x", type=float, default=8.0)
    parser.add_argument("--pub-hz", type=float, default=10.0)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = PhaseCBetaSchedule(goal_x=args.goal_x, pub_hz=args.pub_hz)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
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
