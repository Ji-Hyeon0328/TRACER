#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class WaitPhaseAStatus(Node):
    def __init__(
        self,
        topic: str,
        expect_mode: str,
        expect_vx: Optional[float],
        expect_body_height: Optional[float],
        expect_clearance: Optional[float],
        tol: float,
        timeout_sec: float,
    ) -> None:
        super().__init__("tracer_wait_phase_a_policy_status_v0")
        self.topic = topic
        self.expect_mode = expect_mode
        self.expect_vx = expect_vx
        self.expect_body_height = expect_body_height
        self.expect_clearance = expect_clearance
        self.tol = tol
        self.deadline = time.time() + timeout_sec
        self.last_payload = None
        self.matched_payload = None
        self.create_subscription(String, topic, self.cb, 10)

    def close(self, a: float, b: float) -> bool:
        return math.isfinite(a) and abs(a - b) <= self.tol

    def cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {"raw": msg.data}

        self.last_payload = payload

        if self.expect_mode:
            mode = str(payload.get("mode", ""))
            if mode != self.expect_mode:
                return

        ref = payload.get("mpc_reference", {})
        if not isinstance(ref, dict):
            return

        checks = [
            ("vx", self.expect_vx),
            ("body_height", self.expect_body_height),
            ("swing_clearance", self.expect_clearance),
        ]

        for key, expected in checks:
            if expected is None:
                continue
            try:
                actual = float(ref.get(key))
            except Exception:
                return
            if not self.close(actual, expected):
                return

        self.matched_payload = payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/tracer/highlevel_debug")
    ap.add_argument("--expect-mode", default="")
    ap.add_argument("--expect-vx", type=float, default=None)
    ap.add_argument("--expect-body-height", type=float, default=None)
    ap.add_argument("--expect-clearance", type=float, default=None)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--timeout-sec", type=float, default=5.0)
    args = ap.parse_args()

    rclpy.init()
    node = WaitPhaseAStatus(
        topic=args.topic,
        expect_mode=args.expect_mode,
        expect_vx=args.expect_vx,
        expect_body_height=args.expect_body_height,
        expect_clearance=args.expect_clearance,
        tol=args.tol,
        timeout_sec=args.timeout_sec,
    )

    try:
        while rclpy.ok() and time.time() < node.deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.matched_payload is not None:
                print("[TRACER] matched Phase-A policy status:")
                print(json.dumps(node.matched_payload, indent=2, sort_keys=True))
                return 0

        print("[TRACER][ERR] timeout waiting for Phase-A policy status", file=sys.stderr)
        print("[TRACER][ERR] last payload:", file=sys.stderr)
        print(json.dumps(node.last_payload, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
