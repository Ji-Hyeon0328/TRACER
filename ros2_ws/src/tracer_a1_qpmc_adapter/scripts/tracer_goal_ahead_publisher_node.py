#!/usr/bin/env python3

import math
from typing import Optional, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class TracerGoalAheadPublisherNode(Node):
    """
    Publish /tracer/global_goal as a point ahead of the current robot odom.

    Input:
      /tracer/robot_odom_flat:
        [stamp, x, y, yaw, vx, vy]

    Output:
      /tracer/global_goal:
        [x_goal_world, y_goal_world, enable]

    This is useful because Gazebo/estimator odom may not reset to x=0 after pose reset.
    """

    def __init__(self):
        super().__init__("tracer_goal_ahead_publisher_node")

        self.declare_parameter("odom_topic", "/tracer/robot_odom_flat")
        self.declare_parameter("global_goal_topic", "/tracer/global_goal")
        self.declare_parameter("distance_ahead", 1.5)
        self.declare_parameter("publish_hz", 2.0)
        self.declare_parameter("one_shot", False)
        self.declare_parameter("enable", True)

        self.odom_topic = self.get_parameter("odom_topic").value
        self.global_goal_topic = self.get_parameter("global_goal_topic").value

        self.distance_ahead = float(self.get_parameter("distance_ahead").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.one_shot = bool(self.get_parameter("one_shot").value)
        self.enable = bool(self.get_parameter("enable").value)

        self.last_odom: Optional[List[float]] = None

        self.pub = self.create_publisher(Float64MultiArray, self.global_goal_topic, 10)
        self.create_subscription(Float64MultiArray, self.odom_topic, self.odom_callback, 10)

        self.timer = self.create_timer(
            1.0 / max(0.1, self.publish_hz),
            self.timer_callback,
        )

        self.did_publish = False

        self.get_logger().info(
            f"goal ahead publisher: {self.odom_topic} -> {self.global_goal_topic}, "
            f"distance_ahead={self.distance_ahead:.2f}, one_shot={self.one_shot}"
        )

    def odom_callback(self, msg):
        data = list(msg.data)
        if len(data) < 6:
            self.get_logger().warn(f"robot_odom_flat expects 6 values, got {len(data)}")
            return
        self.last_odom = [float(x) for x in data[:6]]

    def timer_callback(self):
        if self.one_shot and self.did_publish:
            return

        if self.last_odom is None:
            self.get_logger().info("waiting for robot_odom_flat...", throttle_duration_sec=1.0)
            return

        _, x, y, yaw, vx, vy = self.last_odom

        x_goal = x + self.distance_ahead * math.cos(yaw)
        y_goal = y + self.distance_ahead * math.sin(yaw)

        msg = Float64MultiArray()
        msg.data = [float(x_goal), float(y_goal), 1.0 if self.enable else 0.0]
        self.pub.publish(msg)

        self.did_publish = True

        self.get_logger().info(
            f"publish global_goal ahead: robot=({x:.2f},{y:.2f}, yaw={yaw:.2f}) "
            f"goal=({x_goal:.2f},{y_goal:.2f}) d={self.distance_ahead:.2f}",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = TracerGoalAheadPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
