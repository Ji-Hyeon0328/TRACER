#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class WaitMpcReference(Node):
    def __init__(
        self,
        topic: str,
        expect_vx: Optional[float],
        expect_yaw_rate: Optional[float],
        expect_body_height: Optional[float],
        expect_clearance: Optional[float],
        expect_enable: Optional[float],
        tol: float,
        min_matches: int,
        timeout_sec: float,
    ) -> None:
        super().__init__("tracer_wait_phase_a_mpc_reference_v0")

        self.topic = topic
        self.expect_vx = expect_vx
        self.expect_yaw_rate = expect_yaw_rate
        self.expect_body_height = expect_body_height
        self.expect_clearance = expect_clearance
        self.expect_enable = expect_enable
        self.tol = tol
        self.min_matches = max(1, int(min_matches))
        self.deadline = time.time() + timeout_sec

        self.match_count = 0
        self.last_data = None
        self.matched_data = None

        self.create_subscription(Float64MultiArray, topic, self.cb, 10)

    def close(self, actual: float, expected: float) -> bool:
        return math.isfinite(actual) and abs(actual - expected) <= self.tol

    def cb(self, msg: Float64MultiArray) -> None:
        data = list(msg.data)
        self.last_data = data

        # Expected layout:
        # [counter, vx, yaw_rate, body_height, swing_clearance, enable]
        if len(data) < 6:
            self.match_count = 0
            return

        values = {
            "vx": float(data[1]),
            "yaw_rate": float(data[2]),
            "body_height": float(data[3]),
            "swing_clearance": float(data[4]),
            "enable": float(data[5]),
        }

        checks = [
            ("vx", self.expect_vx),
            ("yaw_rate", self.expect_yaw_rate),
            ("body_height", self.expect_body_height),
            ("swing_clearance", self.expect_clearance),
            ("enable", self.expect_enable),
        ]

        ok = True
        for key, expected in checks:
            if expected is None:
                continue
            if not self.close(values[key], expected):
                ok = False
                break

        if ok:
            self.match_count += 1
            self.matched_data = data
        else:
            self.match_count = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/tracer/mpc_reference")
    ap.add_argument("--expect-vx", type=float, default=None)
    ap.add_argument("--expect-yaw-rate", type=float, default=None)
    ap.add_argument("--expect-body-height", type=float, default=None)
    ap.add_argument("--expect-clearance", type=float, default=None)
    ap.add_argument("--expect-enable", type=float, default=None)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--min-matches", type=int, default=5)
    ap.add_argument("--timeout-sec", type=float, default=8.0)
    args = ap.parse_args()

    rclpy.init()
    node = WaitMpcReference(
        topic=args.topic,
        expect_vx=args.expect_vx,
        expect_yaw_rate=args.expect_yaw_rate,
        expect_body_height=args.expect_body_height,
        expect_clearance=args.expect_clearance,
        expect_enable=args.expect_enable,
        tol=args.tol,
        min_matches=args.min_matches,
        timeout_sec=args.timeout_sec,
    )

    try:
        while rclpy.ok() and time.time() < node.deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.match_count >= node.min_matches:
                print("[TRACER] matched direct /tracer/mpc_reference:")
                print(f"  min_matches: {node.min_matches}")
                print(f"  matched_data: {node.matched_data}")
                return 0

        print("[TRACER][ERR] timeout waiting for direct /tracer/mpc_reference", file=sys.stderr)
        print(f"[TRACER][ERR] last_data: {node.last_data}", file=sys.stderr)
        print(f"[TRACER][ERR] match_count: {node.match_count}", file=sys.stderr)
        return 2

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
