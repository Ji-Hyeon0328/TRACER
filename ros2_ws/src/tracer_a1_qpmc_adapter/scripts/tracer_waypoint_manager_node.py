#!/usr/bin/env python3

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class TracerWaypointManagerNode(Node):
    """
    TRACER waypoint mission manager.

    Inputs:
      /tracer/robot_odom_flat:
        [stamp, x_world, y_world, yaw_world, vx_world, vy_world]

      /tracer/waypoints:
        [x0, y0, x1, y1, x2, y2, ...]

      /tracer/mission_cmd:
        [0] stop
        [1] start/resume
        [2] pause
        [3] reset to first waypoint

    Output:
      /tracer/global_goal:
        [x_goal, y_goal, enable]
    """

    def __init__(self):
        super().__init__("tracer_waypoint_manager_node")

        self.declare_parameter("odom_topic", "/tracer/robot_odom_flat")
        self.declare_parameter("waypoints_topic", "/tracer/waypoints")
        self.declare_parameter("mission_cmd_topic", "/tracer/mission_cmd")
        self.declare_parameter("global_goal_topic", "/tracer/global_goal")

        self.declare_parameter("publish_hz", 5.0)
        self.declare_parameter("arrival_distance", 0.30)
        self.declare_parameter("odom_timeout_sec", 1.0)

        # Default rectangular test path.
        self.declare_parameter("default_waypoints", "2.0,0.0;2.0,1.0;0.0,1.0;0.0,0.0")
        self.declare_parameter("auto_start", False)
        self.declare_parameter("loop", False)

        self.odom_topic = self.get_parameter("odom_topic").value
        self.waypoints_topic = self.get_parameter("waypoints_topic").value
        self.mission_cmd_topic = self.get_parameter("mission_cmd_topic").value
        self.global_goal_topic = self.get_parameter("global_goal_topic").value

        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.arrival_distance = float(self.get_parameter("arrival_distance").value)
        self.odom_timeout_sec = float(self.get_parameter("odom_timeout_sec").value)

        self.loop = bool(self.get_parameter("loop").value)

        self.last_odom = None
        self.last_odom_time = None

        self.waypoints: List[Tuple[float, float]] = self.parse_waypoints(
            self.get_parameter("default_waypoints").value
        )
        self.current_idx = 0
        self.active = bool(self.get_parameter("auto_start").value)
        self.finished = False

        self.pub_goal = self.create_publisher(Float64MultiArray, self.global_goal_topic, 10)

        self.create_subscription(
            Float64MultiArray,
            self.odom_topic,
            self.odom_callback,
            10,
        )

        self.create_subscription(
            Float64MultiArray,
            self.waypoints_topic,
            self.waypoints_callback,
            10,
        )

        self.create_subscription(
            Float64MultiArray,
            self.mission_cmd_topic,
            self.mission_cmd_callback,
            10,
        )

        self.timer = self.create_timer(
            1.0 / max(1.0, self.publish_hz),
            self.timer_callback,
        )

        self.get_logger().info(
            f"waypoint manager ready: {len(self.waypoints)} default waypoints, "
            f"auto_start={self.active}, loop={self.loop}"
        )

    def parse_waypoints(self, text: str) -> List[Tuple[float, float]]:
        waypoints = []
        if not text:
            return waypoints

        chunks = text.split(";")
        for chunk in chunks:
            values = [v.strip() for v in chunk.split(",")]
            if len(values) != 2:
                continue
            try:
                waypoints.append((float(values[0]), float(values[1])))
            except ValueError:
                continue

        return waypoints

    def odom_callback(self, msg: Float64MultiArray):
        data = list(msg.data)
        if len(data) < 4:
            self.get_logger().warn(f"odom expects [stamp,x,y,yaw,...], got {len(data)}")
            return

        self.last_odom = data
        self.last_odom_time = self.get_clock().now()

    def waypoints_callback(self, msg: Float64MultiArray):
        data = list(msg.data)

        if len(data) < 2 or len(data) % 2 != 0:
            self.get_logger().warn(
                f"waypoints expects even-length [x0,y0,x1,y1,...], got {len(data)} values"
            )
            return

        self.waypoints = []
        for i in range(0, len(data), 2):
            self.waypoints.append((float(data[i]), float(data[i + 1])))

        self.current_idx = 0
        self.active = True
        self.finished = False

        self.get_logger().info(f"loaded {len(self.waypoints)} waypoints and started mission")

    def mission_cmd_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 1:
            return

        cmd = int(msg.data[0])

        if cmd == 0:
            self.active = False
            self.finished = True
            self.publish_goal(0.0, 0.0, 0.0)
            self.get_logger().info("mission stop")

        elif cmd == 1:
            if self.waypoints:
                self.active = True
                self.finished = False
                self.get_logger().info("mission start/resume")

        elif cmd == 2:
            self.active = False
            self.get_logger().info("mission pause")

        elif cmd == 3:
            self.current_idx = 0
            self.active = True
            self.finished = False
            self.get_logger().info("mission reset to first waypoint")

        else:
            self.get_logger().warn(f"unknown mission cmd: {cmd}")

    def publish_goal(self, x_goal: float, y_goal: float, enable: float):
        msg = Float64MultiArray()
        msg.data = [float(x_goal), float(y_goal), float(enable)]
        self.pub_goal.publish(msg)

    def publish_stop(self, reason: str):
        self.publish_goal(0.0, 0.0, 0.0)
        self.get_logger().info(f"publish stop: {reason}", throttle_duration_sec=1.0)

    def timer_callback(self):
        if self.last_odom is None or self.last_odom_time is None:
            self.publish_stop("waiting for odom")
            return

        now = self.get_clock().now()
        odom_age = (now - self.last_odom_time).nanoseconds * 1e-9

        if odom_age > self.odom_timeout_sec:
            self.publish_stop(f"odom timeout age={odom_age:.2f}s")
            return

        if not self.waypoints:
            self.publish_stop("no waypoints")
            return

        if not self.active or self.finished:
            self.publish_stop("mission inactive/finished")
            return

        _, x, y, yaw = self.last_odom[:4]
        x = float(x)
        y = float(y)

        x_goal, y_goal = self.waypoints[self.current_idx]
        dx = x_goal - x
        dy = y_goal - y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self.arrival_distance:
            self.get_logger().info(
                f"arrived waypoint {self.current_idx}: "
                f"goal=({x_goal:.2f},{y_goal:.2f}) robot=({x:.2f},{y:.2f}) dist={dist:.2f}"
            )

            self.current_idx += 1

            if self.current_idx >= len(self.waypoints):
                if self.loop:
                    self.current_idx = 0
                    self.get_logger().info("looping mission to first waypoint")
                else:
                    self.finished = True
                    self.active = False
                    self.publish_stop("final waypoint reached")
                    return

            x_goal, y_goal = self.waypoints[self.current_idx]

        self.publish_goal(x_goal, y_goal, 1.0)

        self.get_logger().info(
            f"target wp {self.current_idx}/{len(self.waypoints)-1}: "
            f"goal=({x_goal:.2f},{y_goal:.2f}) robot=({x:.2f},{y:.2f},{float(yaw):.2f}) "
            f"dist={dist:.2f}",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = TracerWaypointManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
