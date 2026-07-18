#!/usr/bin/env python3
import os
import re
import math
import argparse
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


@dataclass
class SegmentAction:
    xmin: float
    xmax: float
    vx: float
    h: float
    clearance: float
    name: str


def getenv_float(names, default):
    for name in names:
        v = os.environ.get(name)
        if v is not None and v != "":
            return float(v)
    return float(default)


def parse_schedule(text):
    """
    Format:
      xmin,xmax,vx,h,clearance,name;xmin,xmax,vx,h,clearance,name

    Example:
      0,4,0.205,0.320,0.045,fast_start;4,999,0.200,0.320,0.045,safe_tail
    """
    default = "0,999,0.200,0.320,0.045,base_all"
    text = (text or default).strip()
    out = []

    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) < 5:
            raise ValueError(f"bad schedule chunk: {chunk}")

        xmin = float(parts[0])
        xmax = float(parts[1])
        vx = float(parts[2])
        h = float(parts[3])
        clearance = float(parts[4])
        name = parts[5] if len(parts) >= 6 else f"seg_{len(out)}"

        out.append(SegmentAction(xmin, xmax, vx, h, clearance, name))

    out.sort(key=lambda a: a.xmin)
    return out


class SegmentActionMPCRefNode(Node):
    def __init__(self):
        super().__init__("tracer_phase_d_segment_action_mpc_ref_node_v0")

        self.goal_x = getenv_float(["TRACER_PHASE_C_GOAL_X", "TRACER_GOAL_X"], 8.0)
        self.pub_hz = getenv_float(["TRACER_PHASE_D_PUB_HZ", "TRACER_PUB_HZ"], 10.0)
        self.yaw_rate = getenv_float(["TRACER_PHASE_D_YAW_RATE"], 0.0)
        self.stop_margin = getenv_float(["TRACER_PHASE_D_STOP_MARGIN"], 0.0)

        self.odom_topic = os.environ.get("TRACER_ODOM_TOPIC", "/tracer/robot_odom_flat")
        self.ref_topic = os.environ.get("TRACER_REF_TOPIC", "/tracer/mpc_reference")
        self.label = os.environ.get("TRACER_PHASE_D_SEGMENT_POLICY_LABEL", "segment_policy")

        schedule_text = os.environ.get("TRACER_PHASE_D_SEGMENT_SCHEDULE", "")
        self.schedule = parse_schedule(schedule_text)

        self.seq = 0
        self.x = 0.0
        self.y = 0.0
        self.have_odom = False
        self.stopped = False

        self.pub = self.create_publisher(Float64MultiArray, self.ref_topic, 10)
        self.sub = self.create_subscription(Float64MultiArray, self.odom_topic, self.odom_cb, 10)
        self.timer = self.create_timer(1.0 / self.pub_hz, self.tick)

        self.get_logger().info("Phase-D segment-action MPC ref node started")
        self.get_logger().info(f"label={self.label}, goal_x={self.goal_x:.3f}, pub_hz={self.pub_hz:.1f}")
        self.get_logger().info("schedule=" + ";".join(
            f"{a.xmin:.3f},{a.xmax:.3f},{a.vx:.3f},{a.h:.3f},{a.clearance:.3f},{a.name}"
            for a in self.schedule
        ))

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

    def odom_cb(self, msg):
        self.x, self.y = self.parse_odom_xy(list(msg.data))
        self.have_odom = True

    def select_action(self, x):
        for a in self.schedule:
            if x >= a.xmin and x < a.xmax:
                return a
        return self.schedule[-1]

    def tick(self):
        a = self.select_action(self.x)

        if self.x >= self.goal_x - self.stop_margin:
            self.stopped = True

        if self.stopped:
            vx = 0.0
            enable = 0.0
        else:
            vx = a.vx
            enable = 1.0

        msg = Float64MultiArray()
        msg.data = [
            float(self.seq),
            float(vx),
            float(self.yaw_rate),
            float(a.h),
            float(a.clearance),
            float(enable),
        ]
        self.pub.publish(msg)

        if self.seq % max(1, int(self.pub_hz * 2.0)) == 0:
            self.get_logger().info(
                f"seq={self.seq} x={self.x:.3f} y={self.y:.3f} "
                f"seg={a.name} cmd=[vx={vx:.3f}, yaw={self.yaw_rate:.3f}, "
                f"h={a.h:.3f}, clr={a.clearance:.3f}] stopped={self.stopped}"
            )

        self.seq += 1


def main():
    parser = argparse.ArgumentParser()
    parser.parse_known_args()

    rclpy.init()
    node = SegmentActionMPCRefNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
