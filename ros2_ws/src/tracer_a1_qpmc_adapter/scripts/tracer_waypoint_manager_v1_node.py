#!/usr/bin/env python3

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt(dist2(a, b))


def parse_waypoints(data: List[float]) -> List[Tuple[float, float]]:
    out = []
    n = len(data) // 2
    for i in range(n):
        out.append((float(data[2 * i]), float(data[2 * i + 1])))
    return out


def same_waypoints(a: List[Tuple[float, float]], b: List[Tuple[float, float]], eps=1e-6) -> bool:
    if len(a) != len(b):
        return False
    for p, q in zip(a, b):
        if abs(p[0] - q[0]) > eps or abs(p[1] - q[1]) > eps:
            return False
    return True


class TracerWaypointManagerV1Node(Node):
    """
    Robust waypoint manager.

    Inputs:
      /tracer/waypoints:
        [x0,y0,x1,y1,...] in world/odom frame

      /tracer/mission_cmd:
        [0] stop
        [1] start/resume
        [2] pause
        [3] reset

      /tracer/robot_odom_flat:
        [stamp,x,y,yaw,vx,vy]

    Output:
      /tracer/global_goal:
        [x_goal,y_goal,enable]

    Improvements over simple manager:
      - Does not reset index when the same waypoints are republished.
      - Advances when near waypoint.
      - Advances when waypoint was passed along the segment.
      - Advances when distance started increasing after getting close.
    """

    def __init__(self):
        super().__init__("tracer_waypoint_manager_v1_node")

        self.declare_parameter("waypoints_topic", "/tracer/waypoints")
        self.declare_parameter("mission_cmd_topic", "/tracer/mission_cmd")
        self.declare_parameter("odom_topic", "/tracer/robot_odom_flat")
        self.declare_parameter("global_goal_topic", "/tracer/global_goal")

        self.declare_parameter("publish_hz", 10.0)

        # A1 Gazebo tracking is not precise yet, so start generous.
        self.declare_parameter("reach_radius", 0.55)
        self.declare_parameter("pass_radius", 0.80)
        self.declare_parameter("pass_hysteresis", 0.25)

        self.declare_parameter("advance_on_passed_projection", True)
        self.declare_parameter("auto_start_on_waypoints", False)

        self.waypoints_topic = self.get_parameter("waypoints_topic").value
        self.mission_cmd_topic = self.get_parameter("mission_cmd_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.global_goal_topic = self.get_parameter("global_goal_topic").value

        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.reach_radius = float(self.get_parameter("reach_radius").value)
        self.pass_radius = float(self.get_parameter("pass_radius").value)
        self.pass_hysteresis = float(self.get_parameter("pass_hysteresis").value)
        self.advance_on_passed_projection = bool(self.get_parameter("advance_on_passed_projection").value)
        self.auto_start_on_waypoints = bool(self.get_parameter("auto_start_on_waypoints").value)

        self.waypoints: List[Tuple[float, float]] = []
        self.index = 0
        self.active = False
        self.paused = False

        self.odom: Optional[Tuple[float, float, float]] = None  # x,y,yaw

        self.segment_start: Optional[Tuple[float, float]] = None
        self.min_dist_to_current = float("inf")

        self.pub = self.create_publisher(Float64MultiArray, self.global_goal_topic, 10)

        self.create_subscription(Float64MultiArray, self.waypoints_topic, self.waypoints_callback, 10)
        self.create_subscription(Float64MultiArray, self.mission_cmd_topic, self.mission_callback, 10)
        self.create_subscription(Float64MultiArray, self.odom_topic, self.odom_callback, 10)

        self.timer = self.create_timer(
            1.0 / max(0.1, self.publish_hz),
            self.timer_callback,
        )

        self.get_logger().info(
            f"waypoint manager v1 started. reach_radius={self.reach_radius:.2f}, "
            f"pass_radius={self.pass_radius:.2f}, pass_hysteresis={self.pass_hysteresis:.2f}"
        )

    def current_pos(self) -> Optional[Tuple[float, float]]:
        if self.odom is None:
            return None
        return (self.odom[0], self.odom[1])

    def reset_progress_for_current(self):
        p = self.current_pos()
        if p is not None:
            self.segment_start = p
        else:
            self.segment_start = None
        self.min_dist_to_current = float("inf")

    def waypoints_callback(self, msg):
        new_wps = parse_waypoints(list(msg.data))
        if not new_wps:
            self.get_logger().warn("received empty waypoints")
            return

        # Important: repeated publication of the same waypoints should not reset progress.
        if same_waypoints(new_wps, self.waypoints):
            return

        self.waypoints = new_wps
        self.index = 0
        self.reset_progress_for_current()

        if self.auto_start_on_waypoints:
            self.active = True
            self.paused = False

        self.get_logger().info(
            f"received new waypoints n={len(self.waypoints)}: {self.waypoints}; "
            f"index reset to 0; active={self.active}"
        )

    def mission_callback(self, msg):
        if len(msg.data) < 1:
            return

        cmd = int(msg.data[0])

        if cmd == 0:
            self.active = False
            self.paused = False
            self.get_logger().info("mission stop [0]")

        elif cmd == 1:
            self.active = True
            self.paused = False
            if self.segment_start is None:
                self.reset_progress_for_current()
            self.get_logger().info("mission start/resume [1]")

        elif cmd == 2:
            self.paused = True
            self.get_logger().info("mission pause [2]")

        elif cmd == 3:
            self.index = 0
            self.active = False
            self.paused = False
            self.reset_progress_for_current()
            self.get_logger().info("mission reset [3], index=0, inactive")

        else:
            self.get_logger().warn(f"unknown mission cmd: {cmd}")

    def odom_callback(self, msg):
        data = list(msg.data)
        if len(data) < 4:
            return
        self.odom = (float(data[1]), float(data[2]), float(data[3]))

    def projection_progress(self, p: Tuple[float, float], start: Tuple[float, float], target: Tuple[float, float]) -> float:
        sx, sy = start
        tx, ty = target
        px, py = p

        vx = tx - sx
        vy = ty - sy
        seg_len2 = vx * vx + vy * vy

        if seg_len2 < 1e-9:
            return 1.0

        return ((px - sx) * vx + (py - sy) * vy) / seg_len2

    def should_advance(self, p: Tuple[float, float], target: Tuple[float, float]) -> Tuple[bool, str, float]:
        d = dist(p, target)
        self.min_dist_to_current = min(self.min_dist_to_current, d)

        if d <= self.reach_radius:
            return True, f"reached radius d={d:.2f}", d

        if (
            self.min_dist_to_current <= self.pass_radius
            and d > self.min_dist_to_current + self.pass_hysteresis
        ):
            return True, (
                f"passed after near: d={d:.2f}, "
                f"min_d={self.min_dist_to_current:.2f}"
            ), d

        if self.advance_on_passed_projection and self.segment_start is not None:
            prog = self.projection_progress(p, self.segment_start, target)
            if prog >= 1.0:
                return True, f"passed projection prog={prog:.2f}, d={d:.2f}", d

        return False, "not yet", d

    def advance_waypoint(self, reason: str):
        old_index = self.index
        self.index += 1
        self.reset_progress_for_current()

        if self.index >= len(self.waypoints):
            self.active = False
            self.paused = False
            self.get_logger().info(f"mission complete after waypoint {old_index}: {reason}")
        else:
            self.get_logger().info(
                f"advance waypoint {old_index} -> {self.index}: {reason}; "
                f"next={self.waypoints[self.index]}"
            )

    def publish_goal(self, x: float, y: float, enable: float):
        msg = Float64MultiArray()
        msg.data = [float(x), float(y), float(enable)]
        self.pub.publish(msg)

    def timer_callback(self):
        p = self.current_pos()

        if not self.waypoints or self.index >= len(self.waypoints):
            self.publish_goal(0.0, 0.0, 0.0)
            return

        target = self.waypoints[self.index]

        if p is None:
            self.publish_goal(target[0], target[1], 0.0)
            self.get_logger().info("waiting for odom", throttle_duration_sec=1.0)
            return

        if self.segment_start is None:
            self.segment_start = p

        if not self.active or self.paused:
            self.publish_goal(target[0], target[1], 0.0)
            self.get_logger().info(
                f"inactive/paused index={self.index} target={target}",
                throttle_duration_sec=1.0,
            )
            return

        adv, reason, d = self.should_advance(p, target)
        if adv:
            self.advance_waypoint(reason)

            if self.index >= len(self.waypoints):
                self.publish_goal(0.0, 0.0, 0.0)
                return

            target = self.waypoints[self.index]

        self.publish_goal(target[0], target[1], 1.0)

        self.get_logger().info(
            f"goal index={self.index}/{len(self.waypoints)-1} "
            f"target=({target[0]:.2f},{target[1]:.2f}) "
            f"robot=({p[0]:.2f},{p[1]:.2f}) "
            f"dist={dist(p, target):.2f} min={self.min_dist_to_current:.2f} "
            f"active={self.active}",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = TracerWaypointManagerV1Node()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
