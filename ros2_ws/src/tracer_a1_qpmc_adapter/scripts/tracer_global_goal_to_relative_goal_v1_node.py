#!/usr/bin/env python3

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class TracerGlobalGoalToRelativeGoalV1(Node):
    """
    Robust global-goal to robot-frame relative-goal converter.

    Inputs:
      /tracer/robot_odom_flat:
        [x_rel, y_rel, z_rel, yaw_rel, vx_world, vy_world]

      /tracer/global_goal:
        accepted layouts:
          [x_goal, y_goal, enable]
          [x_goal, y_goal, yaw_goal, enable]

    Output:
      /tracer/relative_goal:
        [x_rel_body, y_rel_body, yaw_rel, enable]
    """

    def __init__(self):
        super().__init__("tracer_global_goal_to_relative_goal_v1_node")

        self.declare_parameter("odom_topic", "/tracer/robot_odom_flat")
        self.declare_parameter("global_goal_topic", "/tracer/global_goal")
        self.declare_parameter("relative_goal_topic", "/tracer/relative_goal")
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("goal_timeout_sec", 2.0)
        self.declare_parameter("odom_timeout_sec", 2.0)
        self.declare_parameter("stop_radius", 0.15)

        self.odom_topic = self.get_parameter("odom_topic").value
        self.global_goal_topic = self.get_parameter("global_goal_topic").value
        self.relative_goal_topic = self.get_parameter("relative_goal_topic").value

        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.goal_timeout_sec = float(self.get_parameter("goal_timeout_sec").value)
        self.odom_timeout_sec = float(self.get_parameter("odom_timeout_sec").value)
        self.stop_radius = float(self.get_parameter("stop_radius").value)

        self.odom: Optional[Tuple[float, float, float]] = None
        self.goal: Optional[Tuple[float, float, float, float]] = None
        self.last_odom_time = None
        self.last_goal_time = None

        self.pub = self.create_publisher(Float64MultiArray, self.relative_goal_topic, 10)

        self.create_subscription(Float64MultiArray, self.odom_topic, self.odom_cb, 10)
        self.create_subscription(Float64MultiArray, self.global_goal_topic, self.goal_cb, 10)

        self.timer = self.create_timer(
            1.0 / max(0.1, self.publish_hz),
            self.timer_cb,
        )

        self.get_logger().info(
            f"global_goal_to_relative_goal_v1 started. "
            f"global layouts: [x,y,enable] or [x,y,yaw,enable]"
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg):
        d = list(msg.data)
        if len(d) < 4:
            self.get_logger().warn(f"odom expects >=4 values, got {len(d)}")
            return

        # /tracer/robot_odom_flat canonical layout:
        #   [x_rel, y_rel, z_rel, yaw_rel, vx_world, vy_world]
        self.odom = (float(d[0]), float(d[1]), float(d[3]))
        self.last_odom_time = self.now_sec()

    def goal_cb(self, msg):
        d = list(msg.data)

        if len(d) >= 4:
            x_goal = float(d[0])
            y_goal = float(d[1])
            yaw_goal = float(d[2])
            enable = float(d[3])
        elif len(d) >= 3:
            x_goal = float(d[0])
            y_goal = float(d[1])
            yaw_goal = 0.0
            enable = float(d[2])
        else:
            self.get_logger().warn(f"global_goal expects 3 or 4 values, got {len(d)}")
            return

        self.goal = (x_goal, y_goal, yaw_goal, enable)
        self.last_goal_time = self.now_sec()

    def publish_relative(self, x_rel, y_rel, yaw_rel, enable):
        msg = Float64MultiArray()
        msg.data = [float(x_rel), float(y_rel), float(yaw_rel), float(enable)]
        self.pub.publish(msg)

    def timer_cb(self):
        now = self.now_sec()

        if self.odom is None:
            self.publish_relative(0.0, 0.0, 0.0, 0.0)
            self.get_logger().info("waiting for odom", throttle_duration_sec=1.0)
            return

        if self.goal is None:
            self.publish_relative(0.0, 0.0, 0.0, 0.0)
            self.get_logger().info("waiting for global goal", throttle_duration_sec=1.0)
            return

        if self.last_odom_time is None or now - self.last_odom_time > self.odom_timeout_sec:
            self.publish_relative(0.0, 0.0, 0.0, 0.0)
            self.get_logger().warn("odom timeout -> relative goal disabled", throttle_duration_sec=1.0)
            return

        if self.last_goal_time is None or now - self.last_goal_time > self.goal_timeout_sec:
            self.publish_relative(0.0, 0.0, 0.0, 0.0)
            self.get_logger().warn("goal timeout -> relative goal disabled", throttle_duration_sec=1.0)
            return

        x, y, yaw = self.odom
        x_goal, y_goal, yaw_goal, enable = self.goal

        if enable <= 0.5:
            self.publish_relative(0.0, 0.0, 0.0, 0.0)
            self.get_logger().info("global goal disabled", throttle_duration_sec=1.0)
            return

        dx = x_goal - x
        dy = y_goal - y

        c = math.cos(yaw)
        s = math.sin(yaw)

        # world -> body frame
        x_rel = c * dx + s * dy
        y_rel = -s * dx + c * dy
        yaw_rel = wrap_to_pi(yaw_goal - yaw)

        d_goal = math.sqrt(x_rel * x_rel + y_rel * y_rel)
        out_enable = 1.0 if d_goal > self.stop_radius else 0.0

        self.publish_relative(x_rel, y_rel, yaw_rel, out_enable)

        self.get_logger().info(
            f"goal=({x_goal:.2f},{y_goal:.2f}, en={enable:.1f}) "
            f"odom=({x:.2f},{y:.2f}, yaw={yaw:.2f}) "
            f"rel=({x_rel:.2f},{y_rel:.2f}, d={d_goal:.2f}, en={out_enable:.1f})",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = TracerGlobalGoalToRelativeGoalV1()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
