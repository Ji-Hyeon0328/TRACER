#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class TracerGlobalGoalToRelativeGoalNode(Node):
    def __init__(self):
        super().__init__("tracer_global_goal_to_relative_goal_node")

        self.declare_parameter("odom_topic", "/tracer/robot_odom_flat")
        self.declare_parameter("global_goal_topic", "/tracer/global_goal")
        self.declare_parameter("relative_goal_topic", "/tracer/relative_goal")
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("odom_timeout_sec", 1.0)
        self.declare_parameter("goal_timeout_sec", 10.0)

        self.odom_topic = self.get_parameter("odom_topic").value
        self.global_goal_topic = self.get_parameter("global_goal_topic").value
        self.relative_goal_topic = self.get_parameter("relative_goal_topic").value
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.odom_timeout_sec = float(self.get_parameter("odom_timeout_sec").value)
        self.goal_timeout_sec = float(self.get_parameter("goal_timeout_sec").value)

        self.last_odom = None
        self.last_odom_time = None
        self.last_goal = None
        self.last_goal_time = None

        self.pub = self.create_publisher(Float64MultiArray, self.relative_goal_topic, 10)

        self.create_subscription(
            Float64MultiArray,
            self.odom_topic,
            self.odom_callback,
            10,
        )

        self.create_subscription(
            Float64MultiArray,
            self.global_goal_topic,
            self.goal_callback,
            10,
        )

        self.timer = self.create_timer(
            1.0 / max(1.0, self.publish_hz),
            self.timer_callback,
        )

        self.get_logger().info(
            f"global goal -> relative goal: {self.global_goal_topic} + {self.odom_topic} -> {self.relative_goal_topic}"
        )

    def odom_callback(self, msg):
        data = list(msg.data)
        if len(data) < 4:
            self.get_logger().warn(f"odom expects [stamp,x,y,yaw,...], got {len(data)}")
            return

        self.last_odom = data
        self.last_odom_time = self.get_clock().now()

    def goal_callback(self, msg):
        data = list(msg.data)
        if len(data) < 2:
            self.get_logger().warn(f"global_goal expects [x_goal,y_goal,enable optional], got {len(data)}")
            return

        self.last_goal = data
        self.last_goal_time = self.get_clock().now()

    def publish_relative_goal(self, x_rel, y_rel, enable):
        msg = Float64MultiArray()
        msg.data = [float(x_rel), float(y_rel), 0.0, float(enable)]
        self.pub.publish(msg)

    def timer_callback(self):
        now = self.get_clock().now()

        if self.last_odom is None or self.last_odom_time is None:
            self.publish_relative_goal(0.0, 0.0, 0.0)
            self.get_logger().info("waiting for odom", throttle_duration_sec=1.0)
            return

        if self.last_goal is None or self.last_goal_time is None:
            self.publish_relative_goal(0.0, 0.0, 0.0)
            self.get_logger().info("waiting for global goal", throttle_duration_sec=1.0)
            return

        odom_age = (now - self.last_odom_time).nanoseconds * 1e-9
        goal_age = (now - self.last_goal_time).nanoseconds * 1e-9

        if odom_age > self.odom_timeout_sec:
            self.publish_relative_goal(0.0, 0.0, 0.0)
            self.get_logger().warn(f"odom timeout age={odom_age:.2f}s", throttle_duration_sec=1.0)
            return

        if goal_age > self.goal_timeout_sec:
            self.publish_relative_goal(0.0, 0.0, 0.0)
            self.get_logger().warn(f"goal timeout age={goal_age:.2f}s", throttle_duration_sec=1.0)
            return

        _, x, y, yaw = self.last_odom[:4]

        x_goal = float(self.last_goal[0])
        y_goal = float(self.last_goal[1])

        enable = 1.0
        if len(self.last_goal) >= 3:
            enable = 1.0 if float(self.last_goal[2]) > 0.5 else 0.0

        dx = x_goal - float(x)
        dy = y_goal - float(y)

        cy = math.cos(float(yaw))
        sy = math.sin(float(yaw))

        # world frame -> robot body frame
        x_rel = cy * dx + sy * dy
        y_rel = -sy * dx + cy * dy

        self.publish_relative_goal(x_rel, y_rel, enable)

        self.get_logger().info(
            f"global=({x_goal:.2f},{y_goal:.2f}) "
            f"robot=({float(x):.2f},{float(y):.2f},{float(yaw):.2f}) "
            f"relative=({x_rel:.2f},{y_rel:.2f}) enable={enable:.1f}",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = TracerGlobalGoalToRelativeGoalNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
