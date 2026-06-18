#!/usr/bin/env python3

import math
from typing import Optional, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def parse_distances_csv(s: str):
    vals = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        vals = [1.0, 2.0, 3.0]
    return vals


class TracerWaypointsAheadPublisherNode(Node):
    """
    Publish /tracer/waypoints ahead of the current robot pose.

    Modes:
      source_mode="odom":
        wait for /tracer/robot_odom_flat = [stamp, x, y, yaw, vx, vy]

      source_mode="gazebo_state" or "fixed_pose":
        use initial_x, initial_y, initial_yaw parameters directly.
        This is useful when Gazebo physics is paused and we do not want to
        unpause only to get an odom sample.

    Outputs:
      /tracer/waypoints = [x0, y0, x1, y1, ...]
      /tracer/mission_cmd = [3] reset, then [1] start
    """

    def __init__(self):
        super().__init__("tracer_waypoints_ahead_publisher_node")

        self.declare_parameter("source_mode", "odom")
        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_yaw", 0.0)

        self.declare_parameter("odom_topic", "/tracer/robot_odom_flat")
        self.declare_parameter("waypoints_topic", "/tracer/waypoints")
        self.declare_parameter("mission_cmd_topic", "/tracer/mission_cmd")

        self.declare_parameter("distances_csv", "1.0,2.0,3.0")
        self.declare_parameter("publish_hz", 2.0)
        self.declare_parameter("publish_duration_sec", 5.0)
        self.declare_parameter("start_mission", True)

        self.source_mode = str(self.get_parameter("source_mode").value)
        self.initial_x = float(self.get_parameter("initial_x").value)
        self.initial_y = float(self.get_parameter("initial_y").value)
        self.initial_yaw = float(self.get_parameter("initial_yaw").value)

        self.odom_topic = self.get_parameter("odom_topic").value
        self.waypoints_topic = self.get_parameter("waypoints_topic").value
        self.mission_cmd_topic = self.get_parameter("mission_cmd_topic").value

        self.distances = parse_distances_csv(self.get_parameter("distances_csv").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.publish_duration_sec = float(self.get_parameter("publish_duration_sec").value)
        self.start_mission = bool(self.get_parameter("start_mission").value)

        self.last_odom: Optional[List[float]] = None
        self.generated_waypoints: Optional[List[float]] = None

        self.pub_waypoints = self.create_publisher(Float64MultiArray, self.waypoints_topic, 10)
        self.pub_mission = self.create_publisher(Float64MultiArray, self.mission_cmd_topic, 10)

        if self.source_mode == "odom":
            self.create_subscription(Float64MultiArray, self.odom_topic, self.odom_callback, 10)
            self.get_logger().info(
                f"waypoints ahead publisher waiting for odom. "
                f"distances={self.distances}, publish_duration={self.publish_duration_sec:.1f}s"
            )
        else:
            self.last_odom = [
                0.0,
                self.initial_x,
                self.initial_y,
                self.initial_yaw,
                0.0,
                0.0,
            ]
            self.get_logger().info(
                f"waypoints ahead publisher using fixed pose source={self.source_mode}: "
                f"x={self.initial_x:.4f}, y={self.initial_y:.4f}, yaw={self.initial_yaw:.4f}, "
                f"distances={self.distances}, publish_duration={self.publish_duration_sec:.1f}s"
            )

        self.start_time = None
        self.did_reset = False
        self.did_start = False
        self.done = False

        self.timer = self.create_timer(
            1.0 / max(0.1, self.publish_hz),
            self.timer_callback,
        )

    def odom_callback(self, msg):
        data = list(msg.data)
        if len(data) < 6:
            self.get_logger().warn(f"robot_odom_flat expects 6 values, got {len(data)}")
            return
        self.last_odom = [float(x) for x in data[:6]]

    def generate_waypoints_once(self):
        if self.generated_waypoints is not None:
            return

        if self.last_odom is None:
            return

        _, x, y, yaw, _, _ = self.last_odom

        wps = []
        for d in self.distances:
            x_goal = x + d * math.cos(yaw)
            y_goal = y + d * math.sin(yaw)
            wps.extend([float(x_goal), float(y_goal)])

        self.generated_waypoints = wps
        self.start_time = self.get_clock().now()

        self.get_logger().info(
            f"generated waypoints from robot=({x:.3f},{y:.3f}, yaw={yaw:.3f}): {wps}"
        )

    def publish_mission_cmd(self, cmd):
        msg = Float64MultiArray()
        msg.data = [float(cmd)]
        self.pub_mission.publish(msg)

    def timer_callback(self):
        if self.done:
            return

        if self.generated_waypoints is None:
            self.generate_waypoints_once()

        if self.generated_waypoints is None:
            self.get_logger().info("waiting for robot_odom_flat...", throttle_duration_sec=1.0)
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

        if elapsed <= self.publish_duration_sec:
            msg = Float64MultiArray()
            msg.data = list(self.generated_waypoints)
            self.pub_waypoints.publish(msg)

            if self.start_mission and not self.did_reset:
                self.publish_mission_cmd(3.0)
                self.did_reset = True
                self.get_logger().info("mission_cmd reset [3]")

            elif self.start_mission and self.did_reset and not self.did_start:
                self.publish_mission_cmd(1.0)
                self.did_start = True
                self.get_logger().info("mission_cmd start [1]")

            self.get_logger().info(
                f"publishing waypoints elapsed={elapsed:.1f}s data={self.generated_waypoints}",
                throttle_duration_sec=1.0,
            )
        else:
            self.get_logger().info("finished publishing waypoints.")
            self.done = True


def main():
    rclpy.init()
    node = TracerWaypointsAheadPublisherNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
