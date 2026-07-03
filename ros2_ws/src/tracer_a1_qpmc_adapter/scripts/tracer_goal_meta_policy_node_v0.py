#!/usr/bin/env python3

import json
import math
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class TracerGoalMetaPolicyNodeV0(Node):
    """
    TRACER Phase-B heuristic goal-conditioned meta-policy.

    Input:
      /tracer/relative_goal:
        [x_rel_body, y_rel_body, yaw_rel, enable]

      /tracer/robot_odom_flat:
        [stamp, x_world, y_world, yaw_world, vx_world, vy_world]

    Output:
      /tracer/mpc_reference:
        [counter, vx_ref, yaw_rate_ref, body_height_ref, swing_clearance_ref, enable]

    Interpretation:
      This is not the final RL policy.
      It is a heuristic Phase-B baseline that occupies the same slot where
      the learned goal-conditioned RL meta-gait planner will later be inserted.
    """

    def __init__(self):
        super().__init__("tracer_goal_meta_policy_node_v0")
        self.reached_latch = False
        self.reached_latch_count = 0

        self.declare_parameter("relative_goal_topic", "/tracer/relative_goal")
        self.declare_parameter("odom_topic", "/tracer/robot_odom_flat")
        self.declare_parameter("mpc_reference_topic", "/tracer/mpc_reference")
        self.declare_parameter("debug_topic", "/tracer/goal_meta_policy_info")

        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("goal_timeout_sec", 1.0)
        self.declare_parameter("odom_timeout_sec", 1.0)

        # Phase-B initial action bounds / defaults
        self.declare_parameter("vx_far", 0.09)
        self.declare_parameter("vx_near", 0.04)
        self.declare_parameter("vx_stop", 0.0)
        self.declare_parameter("yaw_gain", 1.2)
        self.declare_parameter("yaw_rate_max", 0.30)

        self.declare_parameter("body_height", 0.32)
        self.declare_parameter("swing_clearance", 0.04)

        self.declare_parameter("goal_stop_distance", 0.15)
        self.declare_parameter("goal_slow_distance", 0.25)

        # If relative_goal enable becomes 0 near the stop radius, keep holding.
        self.declare_parameter("hold_enable_when_goal_disabled", True)
        self.declare_parameter("stop_enable_on_timeout", False)

        self.relative_goal_topic = str(self.get_parameter("relative_goal_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.mpc_reference_topic = str(self.get_parameter("mpc_reference_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)

        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.goal_timeout_sec = float(self.get_parameter("goal_timeout_sec").value)
        self.odom_timeout_sec = float(self.get_parameter("odom_timeout_sec").value)

        self.counter = 0.0

        self.last_goal: Optional[List[float]] = None
        self.last_goal_time = None

        self.last_odom: Optional[List[float]] = None
        self.last_odom_time = None

        self.pub_mpc = self.create_publisher(Float64MultiArray, self.mpc_reference_topic, 10)
        self.pub_debug = self.create_publisher(String, self.debug_topic, 10)

        self.create_subscription(
            Float64MultiArray,
            self.relative_goal_topic,
            self.goal_cb,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            self.odom_topic,
            self.odom_cb,
            10,
        )

        self.timer = self.create_timer(
            1.0 / max(1.0, self.publish_hz),
            self.timer_cb,
        )

        self.get_logger().info(
            "TRACER Phase-B goal meta-policy v0 started: "
            f"{self.relative_goal_topic} + {self.odom_topic} -> {self.mpc_reference_topic}"
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def goal_cb(self, msg: Float64MultiArray):
        data = [float(x) for x in list(msg.data)]
        if len(data) < 2:
            self.get_logger().warn(f"relative_goal expects >=2 values, got {len(data)}")
            return

        while len(data) < 4:
            data.append(1.0)

        self.last_goal = data[:4]
        self.last_goal_time = self.now_sec()

    def odom_cb(self, msg: Float64MultiArray):
        data = [float(x) for x in list(msg.data)]
        if len(data) < 6:
            self.get_logger().warn(f"robot_odom_flat expects >=6 values, got {len(data)}")
            return

        self.last_odom = data[:6]
        self.last_odom_time = self.now_sec()

    def publish_mpc(self, vx: float, yaw_rate: float, body_h: float, clearance: float, enable: float):
        msg = Float64MultiArray()
        msg.data = [
            float(self.counter),
            float(vx),
            float(yaw_rate),
            float(body_h),
            float(clearance),
            float(enable),
        ]
        self.pub_mpc.publish(msg)

    def publish_debug(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.pub_debug.publish(msg)

    def publish_stop(self, reason: str, enable: float):
        body_h = float(self.get_parameter("body_height").value)
        clearance = float(self.get_parameter("swing_clearance").value)

        self.publish_mpc(0.0, 0.0, body_h, clearance, enable)
        self.publish_debug({
            "mode": "stop",
            "reason": reason,
            "counter": self.counter,
            "vx": 0.0,
            "yaw_rate": 0.0,
            "body_height": body_h,
            "swing_clearance": clearance,
            "enable": enable,
        })

        self.get_logger().info(
            f"Phase-B goal meta-policy stop: reason={reason}, enable={enable:.1f}",
            throttle_duration_sec=1.0,
        )

    def compute_action(self, goal: List[float]):
        x_rel = float(goal[0])
        y_rel = float(goal[1])
        yaw_rel = float(goal[2]) if len(goal) >= 3 else 0.0
        goal_enable = float(goal[3]) if len(goal) >= 4 else 1.0

        dist = math.sqrt(x_rel * x_rel + y_rel * y_rel)
        heading_error = math.atan2(y_rel, max(1.0e-6, x_rel))

        vx_far = float(self.get_parameter("vx_far").value)
        vx_near = float(self.get_parameter("vx_near").value)
        vx_stop = float(self.get_parameter("vx_stop").value)
        yaw_gain = float(self.get_parameter("yaw_gain").value)
        yaw_rate_max = abs(float(self.get_parameter("yaw_rate_max").value))

        body_h = float(self.get_parameter("body_height").value)
        clearance = float(self.get_parameter("swing_clearance").value)

        stop_d = float(self.get_parameter("goal_stop_distance").value)
        slow_d = float(self.get_parameter("goal_slow_distance").value)
        hold_enable_when_goal_disabled = bool(
            self.get_parameter("hold_enable_when_goal_disabled").value
        )

        if goal_enable <= 0.5:
            # In Phase-B, relative_goal_v1 may set enable=0 inside stop_radius.
            # Treat this as "hold at goal" rather than killing the controller.
            enable = 1.0 if hold_enable_when_goal_disabled else 0.0
            return {
                "mode": "goal_disabled_hold",
                "x_rel": x_rel,
                "y_rel": y_rel,
                "yaw_rel": yaw_rel,
                "dist": dist,
                "heading_error": heading_error,
                "vx": 0.0,
                "yaw_rate": 0.0,
                "body_height": body_h,
                "swing_clearance": clearance,
                "enable": enable,
            }

        if dist <= stop_d:
            mode = "goal_reached_hold"
            vx = vx_stop
            yaw_rate = 0.0
            enable = 1.0
        elif dist <= slow_d:
            mode = "near_goal_slow"
            vx = vx_near
            yaw_rate = clamp(yaw_gain * heading_error, -yaw_rate_max, yaw_rate_max)
            enable = 1.0
        else:
            mode = "go_to_goal"
            vx = vx_far
            yaw_rate = clamp(yaw_gain * heading_error, -yaw_rate_max, yaw_rate_max)
            enable = 1.0

        return {
            "mode": mode,
            "x_rel": x_rel,
            "y_rel": y_rel,
            "yaw_rel": yaw_rel,
            "dist": dist,
            "heading_error": heading_error,
            "vx": vx,
            "yaw_rate": yaw_rate,
            "body_height": body_h,
            "swing_clearance": clearance,
            "enable": enable,
        }

    def timer_cb(self):
        self.counter += 1.0
        now = self.now_sec()

        stop_enable_on_timeout = bool(self.get_parameter("stop_enable_on_timeout").value)
        timeout_enable = 0.0 if stop_enable_on_timeout else 1.0

        if self.last_goal is None or self.last_goal_time is None:
            self.publish_stop("waiting_for_relative_goal", timeout_enable)
            return

        goal_age = now - self.last_goal_time
        if goal_age > self.goal_timeout_sec:
            self.publish_stop(f"relative_goal_timeout_{goal_age:.2f}s", timeout_enable)
            return

        odom_fresh = False
        odom_age = None
        if self.last_odom is not None and self.last_odom_time is not None:
            odom_age = now - self.last_odom_time
            odom_fresh = odom_age <= self.odom_timeout_sec

        action = self.compute_action(self.last_goal)

        self.publish_mpc(
            action["vx"],
            action["yaw_rate"],
            action["body_height"],
            action["swing_clearance"],
            action["enable"],
        )

        action["counter"] = self.counter
        action["goal_age_sec"] = goal_age
        action["odom_fresh"] = odom_fresh
        action["odom_age_sec"] = odom_age
        action["policy_type"] = "heuristic_phase_b_v0"
        action["action_layout"] = "[vx_ref,yaw_rate_ref,body_height_ref,swing_clearance_ref]"

        self.publish_debug(action)

        self.get_logger().info(
            f"Phase-B {action['mode']}: "
            f"d={action['dist']:.3f} heading={action['heading_error']:.3f} "
            f"-> vx={action['vx']:.3f} yaw={action['yaw_rate']:.3f} "
            f"h={action['body_height']:.3f} clr={action['swing_clearance']:.3f} "
            f"en={action['enable']:.1f}",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = TracerGoalMetaPolicyNodeV0()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
