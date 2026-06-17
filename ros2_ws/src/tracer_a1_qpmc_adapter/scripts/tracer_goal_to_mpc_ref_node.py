#!/usr/bin/env python3

import math
from typing import List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class TracerGoalToMpcRefNode(Node):
    """
    TRACER first closed-loop outer-loop node.

    Input:
      /tracer/relative_goal : Float64MultiArray
        data[0] = x_rel [m] in robot/body frame
        data[1] = y_rel [m] in robot/body frame
        data[2] = optional yaw_rel [rad], currently not required
        data[3] = optional enable override, >0.5 means enable

    Output:
      /tracer/mpc_reference : Float64MultiArray
        data[0] = counter
        data[1] = v_x [m/s]
        data[2] = yaw_rate [rad/s]
        data[3] = body_height [m]
        data[4] = clearance [m]
        data[5] = enable_walking
    """

    def __init__(self):
        super().__init__("tracer_goal_to_mpc_ref_node")

        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("goal_topic", "/tracer/relative_goal")
        self.declare_parameter("mpc_ref_topic", "/tracer/mpc_reference")

        self.declare_parameter("kx", 0.35)
        self.declare_parameter("kyaw", 1.2)

        self.declare_parameter("vx_min", 0.0)
        self.declare_parameter("vx_max", 0.22)
        self.declare_parameter("yaw_rate_max", 0.45)

        self.declare_parameter("goal_stop_distance", 0.18)
        self.declare_parameter("yaw_slowdown_angle", 0.9)

        self.declare_parameter("body_height", 0.30)
        self.declare_parameter("clearance", 0.03)

        self.declare_parameter("goal_timeout_sec", 1.0)

        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.goal_topic = self.get_parameter("goal_topic").value
        self.mpc_ref_topic = self.get_parameter("mpc_ref_topic").value

        self.kx = float(self.get_parameter("kx").value)
        self.kyaw = float(self.get_parameter("kyaw").value)

        self.vx_min = float(self.get_parameter("vx_min").value)
        self.vx_max = float(self.get_parameter("vx_max").value)
        self.yaw_rate_max = float(self.get_parameter("yaw_rate_max").value)

        self.goal_stop_distance = float(self.get_parameter("goal_stop_distance").value)
        self.yaw_slowdown_angle = float(self.get_parameter("yaw_slowdown_angle").value)

        self.body_height = float(self.get_parameter("body_height").value)
        self.clearance = float(self.get_parameter("clearance").value)

        self.goal_timeout_sec = float(self.get_parameter("goal_timeout_sec").value)

        self.counter = 0.0
        self.last_goal = None
        self.last_goal_time = None

        self.pub = self.create_publisher(Float64MultiArray, self.mpc_ref_topic, 10)
        self.sub = self.create_subscription(
            Float64MultiArray,
            self.goal_topic,
            self.goal_callback,
            10,
        )

        period = 1.0 / max(1.0, self.publish_hz)
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f"TRACER goal tracker: {self.goal_topic} -> {self.mpc_ref_topic}"
        )

    def goal_callback(self, msg: Float64MultiArray):
        data = list(msg.data)
        if len(data) < 2:
            self.get_logger().warn(
                f"relative_goal expects at least [x_rel, y_rel], got {len(data)} values"
            )
            return

        self.last_goal = data
        self.last_goal_time = self.get_clock().now()

    def compute_reference(self, goal: List[float]):
        x_rel = float(goal[0])
        y_rel = float(goal[1])

        dist = math.sqrt(x_rel * x_rel + y_rel * y_rel)
        yaw_error = math.atan2(y_rel, max(1e-6, x_rel))

        goal_enable_override = None
        if len(goal) >= 4:
            goal_enable_override = float(goal[3]) > 0.5

        if dist < self.goal_stop_distance:
            vx = 0.0
            yaw_rate = 0.0
            enable = 0.0
        else:
            yaw_alignment = max(
                0.0,
                1.0 - abs(yaw_error) / max(1e-6, self.yaw_slowdown_angle),
            )

            vx_raw = self.kx * x_rel
            vx = clamp(vx_raw, self.vx_min, self.vx_max) * yaw_alignment

            yaw_rate = clamp(
                self.kyaw * yaw_error,
                -self.yaw_rate_max,
                self.yaw_rate_max,
            )

            enable = 1.0

        if goal_enable_override is not None:
            enable = 1.0 if goal_enable_override else 0.0
            if not goal_enable_override:
                vx = 0.0
                yaw_rate = 0.0

        return vx, yaw_rate, self.body_height, self.clearance, enable, dist, yaw_error

    def publish_stop(self, reason: str):
        msg = Float64MultiArray()
        msg.data = [
            self.counter,
            0.0,
            0.0,
            self.body_height,
            self.clearance,
            0.0,
        ]
        self.pub.publish(msg)

        self.get_logger().info(
            f"publish stop: {reason}",
            throttle_duration_sec=1.0,
        )

    def timer_callback(self):
        self.counter += 1.0

        if self.last_goal is None or self.last_goal_time is None:
            self.publish_stop("no goal")
            return

        age = (self.get_clock().now() - self.last_goal_time).nanoseconds * 1e-9
        if age > self.goal_timeout_sec:
            self.publish_stop(f"goal timeout age={age:.2f}s")
            return

        vx, yaw_rate, body_height, clearance, enable, dist, yaw_error = self.compute_reference(
            self.last_goal
        )

        msg = Float64MultiArray()
        msg.data = [
            self.counter,
            vx,
            yaw_rate,
            body_height,
            clearance,
            enable,
        ]
        self.pub.publish(msg)

        self.get_logger().info(
            f"goal dist={dist:.2f} yaw_err={yaw_error:.2f} -> "
            f"vx={vx:.3f} yaw_rate={yaw_rate:.3f} h={body_height:.3f} "
            f"clr={clearance:.3f} enable={enable:.1f}",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = TracerGoalToMpcRefNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
