#!/usr/bin/env python3
import os
import json
import time
from typing import Dict, Tuple

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray


Theta = Tuple[float, float, float]  # vx, body_h, clearance


def parse_action_table(text: str) -> Dict[str, Theta]:
    """
    Format:
      label:vx,body_h,clearance;label:vx,body_h,clearance

    Example:
      flat:0.210,0.320,0.045;rough:0.210,0.320,0.045;downslope:0.2025,0.320,0.045
    """
    table: Dict[str, Theta] = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"bad action table item: {item}")
        key, vals = item.split(":", 1)
        parts = [float(x.strip()) for x in vals.split(",")]
        if len(parts) != 3:
            raise ValueError(f"bad theta for {key}: {vals}")
        table[key.strip()] = (parts[0], parts[1], parts[2])
    return table


class ContextMetaSelector(Node):
    def __init__(self):
        super().__init__("tracer_phase_d4_context_meta_selector_node_v0")

        self.goal_x = float(os.environ.get("TRACER_PHASE_D4_GOAL_X", "8.0"))
        self.stop_margin = float(os.environ.get("TRACER_PHASE_D4_STOP_MARGIN", "0.10"))
        self.lateral_bound = float(os.environ.get("TRACER_PHASE_D4_LATERAL_BOUND", "2.0"))
        self.yaw_k = float(os.environ.get("TRACER_PHASE_D4_YAW_K", "0.20"))
        self.yaw_max = float(os.environ.get("TRACER_PHASE_D4_YAW_MAX", "0.35"))
        self.yaw_sign = float(os.environ.get("TRACER_PHASE_D4_YAW_SIGN", "0.0"))
        self.startup_timeout_s = float(os.environ.get("TRACER_PHASE_D4_STARTUP_TIMEOUT_S", "60.0"))
        self.startup_min_x = float(os.environ.get("TRACER_PHASE_D4_STARTUP_MIN_X", "0.20"))
        self.hold_vx = float(os.environ.get("TRACER_PHASE_D4_HOLD_VX", "0.025"))
        self.t0 = time.time()
        self.startup_failed = False
        self.pub_hz = float(os.environ.get("TRACER_PHASE_D4_PUB_HZ", "10.0"))

        self.default_label = os.environ.get("TRACER_PHASE_D4_DEFAULT_CONTEXT", "flat")
        # Rollout-selected D4.2 default table: rough_clearance.
        # rough terrain uses slightly lower vx and higher swing clearance.
        default_table = (
            "flat:0.210,0.320,0.045;"
            "start_flat:0.210,0.320,0.045;"
            "rough:0.2050,0.320,0.055;"
            "upslope:0.210,0.320,0.045;"
            "downslope:0.2025,0.320,0.045;"
            "goal_flat:0.2025,0.320,0.045;"
            "unknown:0.2025,0.320,0.045"
        )
        self.policy_json = os.environ.get("TRACER_PHASE_D4_POLICY_JSON", "").strip()
        self.policy_name = "env_or_default_table"

        action_table_text = os.environ.get("TRACER_PHASE_D4_CONTEXT_ACTION_TABLE", default_table)

        if self.policy_json:
            with open(self.policy_json, "r") as f:
                policy = json.load(f)

            self.policy_name = policy.get("policy_name", self.policy_json)

            if "action_table" in policy:
                action_table_text = policy["action_table"]

            # Explicit env var still has priority. Otherwise use the policy artifact hold_vx.
            if "TRACER_PHASE_D4_HOLD_VX" not in os.environ and "hold_vx" in policy:
                self.hold_vx = float(policy["hold_vx"])

        self.action_table = parse_action_table(action_table_text)

        self.current_context = self.default_label
        self.x = 0.0
        self.y = 0.0
        self.seq = 0
        self.stopped = False

        self.ref_topic = os.environ.get("TRACER_REF_TOPIC", "/tracer/mpc_reference")
        self.pub = self.create_publisher(Float64MultiArray, self.ref_topic, 10)
        self.create_subscription(String, "/tracer/terrain_context_label", self.on_context, 10)
        self.create_subscription(Float64MultiArray, "/tracer/robot_odom_flat", self.on_odom, 10)

        self.timer = self.create_timer(1.0 / self.pub_hz, self.on_timer)

        self.get_logger().info("Phase-D4 context meta selector started")
        self.get_logger().info(
            f"goal_x={self.goal_x}, stop_margin={self.stop_margin}, "
            f"lateral_bound={self.lateral_bound}, yaw_k={self.yaw_k}, "
            f"yaw_max={self.yaw_max}, yaw_sign={self.yaw_sign}, hold_vx={self.hold_vx}"
        )
        self.get_logger().info(f"default_context={self.default_label}")
        self.get_logger().info(f"policy_name={self.policy_name}")
        self.get_logger().info(f"policy_json={self.policy_json if self.policy_json else 'none'}")
        self.get_logger().info(f"action_table={self.action_table}")

    def on_context(self, msg: String):
        label = msg.data.strip()
        if label:
            self.current_context = label

    def parse_odom_xy(self, data):
        """
        Supports both likely layouts:
          [x,y,z,roll,pitch,yaw,...]
          [seq,stamp,x,y,z,roll,pitch,yaw,...]
        """
        if len(data) >= 4 and (abs(data[0]) > 20.0 or abs(data[1]) > 1000.0):
            return float(data[2]), float(data[3])
        if len(data) >= 2:
            return float(data[0]), float(data[1])
        return self.x, self.y

    def on_odom(self, msg: Float64MultiArray):
        self.x, self.y = self.parse_odom_xy(list(msg.data))

    def select_theta(self) -> Theta:
        label = self.current_context
        if label in self.action_table:
            return self.action_table[label]
        if "unknown" in self.action_table:
            return self.action_table["unknown"]
        return self.action_table.get(self.default_label, (0.2025, 0.320, 0.045))

    def on_timer(self):
        self.seq += 1

        out_of_lane = abs(self.y) > self.lateral_bound
        elapsed = time.time() - self.t0
        startup_failed = elapsed > self.startup_timeout_s and self.x < self.startup_min_x
        if startup_failed:
            self.startup_failed = True

        if self.x >= self.goal_x - self.stop_margin:
            self.stopped = True

        if out_of_lane or startup_failed:
            self.stopped = True

        if self.stopped:
            vx, body_h, clearance = self.hold_vx, 0.320, 0.045
            yaw_rate = 0.0
            # Keep controller enabled during stop/hold.
            # A small positive hold_vx can compensate backward drift near/after the goal region.
            enable = 1.0
        else:
            vx, body_h, clearance = self.select_theta()
            yaw_rate = self.yaw_sign * self.yaw_k * self.y
            if yaw_rate > self.yaw_max:
                yaw_rate = self.yaw_max
            if yaw_rate < -self.yaw_max:
                yaw_rate = -self.yaw_max
            enable = 1.0

        msg = Float64MultiArray()
        msg.data = [
            float(self.seq),
            float(vx),
            float(yaw_rate),
            float(body_h),
            float(clearance),
            float(enable),
        ]
        self.pub.publish(msg)

        if self.seq % int(max(1.0, self.pub_hz)) == 0:
            self.get_logger().info(
                f"seq={self.seq} context={self.current_context} "
                f"x={self.x:.3f} y={self.y:.3f} "
                f"cmd=[seq={self.seq}, vx={vx:.4f}, yaw={yaw_rate:.4f}, h={body_h:.4f}, clr={clearance:.4f}, enable={enable:.1f}] "
                f"out_of_lane={out_of_lane} startup_failed={self.startup_failed} stopped={self.stopped}"
            )


def main():
    rclpy.init()
    node = ContextMetaSelector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
