#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


LEVEL_TO_CODE = {
    "stable": 0.0,
    "caution": 1.0,
    "unstable": 2.0,
    "unknown": -1.0,
}

ACTION_TO_CODE = {
    "keep": 0.0,
    "would_cautious": 1.0,
    "would_conservative_probe": 2.0,
    "prior_avoid_keep": 3.0,
    "prior_recovery_keep": 4.0,
    "no_valid_keep": 5.0,
    "failed_candidate_keep": 6.0,
    "unknown": -1.0,
}


class FakeRamGateAdvicePublisher(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("tracer_fake_ram_gate_advice_publisher")
        self.args = args
        self.pub = self.create_publisher(Float64MultiArray, args.topic, 10)

    def make_msg(self) -> Float64MultiArray:
        data = [0.0] * 16

        # Keep this schema aligned with tracer_fusion_policy_mpc_ref_node.py:on_ram_gate_advice()
        data[2] = LEVEL_TO_CODE[self.args.level]
        data[3] = ACTION_TO_CODE[self.args.action]
        data[4] = float(self.args.would_override)
        data[5] = float(self.args.control_risk)
        data[6] = float(self.args.ctrl_ema)
        data[7] = float(self.args.future_risk)
        data[8] = float(self.args.fallen_prob)
        data[9] = float(self.args.recovery_prob)
        data[10] = float(self.args.sigma_mean)
        data[11] = float(self.args.rho_norm)
        data[12] = float(self.args.style_score)
        data[13] = float(self.args.gate_vx_scale)
        data[14] = float(self.args.gate_body_height_delta)
        data[15] = float(self.args.gate_clearance_delta)

        msg = Float64MultiArray()
        msg.data = data
        return msg

    def run(self) -> None:
        msg = self.make_msg()
        dt = 1.0 / max(1e-6, float(self.args.hz))
        t_end = time.time() + max(0.0, float(self.args.duration))

        self.get_logger().info(
            f"publishing fake RAM gate advice topic={self.args.topic} "
            f"level={self.args.level} action={self.args.action} "
            f"would_override={self.args.would_override:.3f} "
            f"control_risk={self.args.control_risk:.3f} "
            f"fallen={self.args.fallen_prob:.3f} "
            f"recovery={self.args.recovery_prob:.3f} "
            f"hz={self.args.hz:.2f} duration={self.args.duration:.2f}s "
            f"data={list(msg.data)}"
        )

        # Give subscribers a short discovery window.
        time.sleep(float(self.args.discovery_wait))

        count = 0
        while rclpy.ok() and time.time() <= t_end:
            self.pub.publish(msg)
            count += 1
            time.sleep(dt)

        self.get_logger().info(f"published {count} fake RAM gate messages")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default="/tracer/ram_gate_advice")
    p.add_argument("--level", choices=sorted(LEVEL_TO_CODE), default="unstable")
    p.add_argument("--action", choices=sorted(ACTION_TO_CODE), default="would_conservative_probe")
    p.add_argument("--would-override", type=float, default=1.0)
    p.add_argument("--control-risk", type=float, default=0.80)
    p.add_argument("--ctrl-ema", type=float, default=0.80)
    p.add_argument("--future-risk", type=float, default=0.80)
    p.add_argument("--fallen-prob", type=float, default=0.50)
    p.add_argument("--recovery-prob", type=float, default=0.70)
    p.add_argument("--sigma-mean", type=float, default=0.0)
    p.add_argument("--rho-norm", type=float, default=0.0)
    p.add_argument("--style-score", type=float, default=0.0)
    p.add_argument("--gate-vx-scale", type=float, default=1.0)
    p.add_argument("--gate-body-height-delta", type=float, default=0.0)
    p.add_argument("--gate-clearance-delta", type=float, default=0.0)
    p.add_argument("--hz", type=float, default=5.0)
    p.add_argument("--duration", type=float, default=2.0)
    p.add_argument("--discovery-wait", type=float, default=0.4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = FakeRamGateAdvicePublisher(args)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
