#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class CaptureHighlevelDebugOnce(Node):
    def __init__(self):
        super().__init__("tracer_capture_highlevel_debug_once")
        self.out = Path(os.environ.get(
            "TRACER_HIGHLEVEL_DEBUG_CAPTURE",
            "data/debug/highlevel_debug_once.json",
        ))
        self.out.parent.mkdir(parents=True, exist_ok=True)

        self.sub = self.create_subscription(
            String,
            "/tracer/highlevel_debug",
            self.cb,
            10,
        )
        self.get_logger().info(f"waiting for /tracer/highlevel_debug -> {self.out}")

    def cb(self, msg: String):
        raw = msg.data

        try:
            obj = json.loads(raw)
            self.out.write_text(json.dumps(obj, indent=2))
            self.get_logger().info(f"wrote JSON debug message: {self.out}")
        except Exception:
            self.out.write_text(raw)
            self.get_logger().warn(f"wrote raw debug message: {self.out}")

        os._exit(0)


def main():
    rclpy.init()
    node = CaptureHighlevelDebugOnce()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
