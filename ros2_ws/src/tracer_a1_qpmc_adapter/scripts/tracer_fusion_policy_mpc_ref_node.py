#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def find_default_policy() -> Path:
    here = Path(__file__).resolve()
    # .../TRACER/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/file.py
    root = here.parents[4]
    return root / "configs/highlevel_policy/tracer_fusion_policy_v0.json"


class TracerFusionPolicyMpcRefNode(Node):
    def __init__(self):
        super().__init__("tracer_fusion_policy_mpc_ref_node")

        default_policy = str(find_default_policy())

        self.declare_parameter("policy_json", os.environ.get("TRACER_FUSION_POLICY_JSON", default_policy))
        self.declare_parameter("terrain", os.environ.get("TRACER_TERRAIN_KEY", "flat_normal"))
        self.declare_parameter("hz", float(os.environ.get("TRACER_MPC_REF_HZ", "10.0")))
        self.declare_parameter("duration_sec", float(os.environ.get("TRACER_PUBLISH_DURATION", "0.0")))

        # Optional terrain-aware command transition ramp.
        # Used for slippery active fallback experiments.
        self.declare_parameter("ramp_body_height_enable", int(os.environ.get("TRACER_RAMP_BODY_HEIGHT_ENABLE", "0")))
        self.declare_parameter("ramp_body_height_start", float(os.environ.get("TRACER_RAMP_BODY_HEIGHT_START", "0.325")))
        self.declare_parameter("ramp_body_height_duration", float(os.environ.get("TRACER_RAMP_BODY_HEIGHT_DURATION", "2.0")))

        # Optional velocity ramp.
        # This lets the robot complete the body-height transition first,
        # then gradually apply a small micro-brake / backstep velocity.
        self.declare_parameter("ramp_vx_enable", int(os.environ.get("TRACER_RAMP_VX_ENABLE", "0")))
        self.declare_parameter("ramp_vx_start", float(os.environ.get("TRACER_RAMP_VX_START", "0.0")))
        self.declare_parameter("ramp_vx_delay", float(os.environ.get("TRACER_RAMP_VX_DELAY", "0.0")))
        self.declare_parameter("ramp_vx_duration", float(os.environ.get("TRACER_RAMP_VX_DURATION", "2.0")))

        self.policy_json = Path(self.get_parameter("policy_json").value)
        self.terrain = str(self.get_parameter("terrain").value)
        self.hz = float(self.get_parameter("hz").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.ramp_body_height_enable = bool(int(self.get_parameter("ramp_body_height_enable").value))
        self.ramp_body_height_start = float(self.get_parameter("ramp_body_height_start").value)
        self.ramp_body_height_duration = max(1e-6, float(self.get_parameter("ramp_body_height_duration").value))

        self.ramp_vx_enable = bool(int(self.get_parameter("ramp_vx_enable").value))
        self.ramp_vx_start = float(self.get_parameter("ramp_vx_start").value)
        self.ramp_vx_delay = max(0.0, float(self.get_parameter("ramp_vx_delay").value))
        self.ramp_vx_duration = max(1e-6, float(self.get_parameter("ramp_vx_duration").value))

        if self.hz <= 0.0:
            raise RuntimeError("hz must be positive")

        if not self.policy_json.exists():
            raise FileNotFoundError(self.policy_json)

        self.policy = json.loads(self.policy_json.read_text())
        terrains = self.policy.get("terrains", {})

        if self.terrain not in terrains:
            keys = "\n".join(sorted(terrains.keys()))
            raise RuntimeError(
                f"Unknown terrain key: {self.terrain}\n"
                f"Available terrain keys:\n{keys}"
            )

        self.entry = terrains[self.terrain]
        self.command = self.entry["command"]
        self.counter = 0.0
        self.t0 = time.time()
        self.last_print = 0.0

        self.pub = self.create_publisher(Float64MultiArray, "/tracer/mpc_reference", 10)
        self.beta_pub = self.create_publisher(Float64MultiArray, "/tracer/objective_weights", 10)
        self.timer = self.create_timer(1.0 / self.hz, self.on_timer)

        self.get_logger().info(f"policy_json={self.policy_json}")
        self.get_logger().info(
            f"terrain={self.terrain} "
            f"mode={self.entry['fused_mode']} "
            f"style={self.entry['suggested_style']} "
            f"risk={self.entry['fused_risk']:.3f}"
        )
        self.get_logger().info(
            f"command vx={self.command['vx']:.3f} "
            f"yaw={self.command['yaw_rate']:.3f} "
            f"h={self.command['body_height']:.3f} "
            f"clr={self.command['swing_clearance']:.3f} "
            f"enable={self.command['enable']:.1f}"
        )
        if self.ramp_body_height_enable:
            self.get_logger().info(
                f"body-height ramp enabled: "
                f"start_h={self.ramp_body_height_start:.3f} "
                f"target_h={self.command['body_height']:.3f} "
                f"duration={self.ramp_body_height_duration:.3f}s"
            )

        if self.ramp_vx_enable:
            self.get_logger().info(
                f"vx ramp enabled: "
                f"start_vx={self.ramp_vx_start:.4f} "
                f"target_vx={self.command['vx']:.4f} "
                f"delay={self.ramp_vx_delay:.3f}s "
                f"duration={self.ramp_vx_duration:.3f}s"
            )

    def on_timer(self):
        now = time.time()
        elapsed = now - self.t0

        if self.duration_sec > 0.0 and elapsed > self.duration_sec:
            self.get_logger().info("finished publishing fusion policy command")
            rclpy.shutdown()
            return

        msg = Float64MultiArray()
        body_height = float(self.command["body_height"])
        if self.ramp_body_height_enable:
            alpha = min(1.0, max(0.0, elapsed / self.ramp_body_height_duration))
            body_height = (
                (1.0 - alpha) * self.ramp_body_height_start
                + alpha * float(self.command["body_height"])
            )

        vx = float(self.command["vx"])
        if self.ramp_vx_enable:
            if elapsed < self.ramp_vx_delay:
                vx = self.ramp_vx_start
            else:
                vx_elapsed = elapsed - self.ramp_vx_delay
                beta = min(1.0, max(0.0, vx_elapsed / self.ramp_vx_duration))
                vx = (
                    (1.0 - beta) * self.ramp_vx_start
                    + beta * float(self.command["vx"])
                )

        msg.data = [
            float(self.counter),
            float(vx),
            float(self.command["yaw_rate"]),
            float(body_height),
            float(self.command["swing_clearance"]),
            float(self.command["enable"]),
        ]
        self.pub.publish(msg)

        beta = self.entry.get("beta", {})
        beta_msg = Float64MultiArray()
        beta_msg.data = [
            float(beta.get("motion", 1.0 / 3.0)),
            float(beta.get("stability", 1.0 / 3.0)),
            float(beta.get("energy", 1.0 / 3.0)),
        ]
        self.beta_pub.publish(beta_msg)

        self.counter += 1.0

        if now - self.last_print >= 1.0:
            self.last_print = now
            self.get_logger().info(
                f"publishing /tracer/mpc_reference "
                f"terrain={self.terrain} "
                f"style={self.entry['suggested_style']} "
                f"data={msg.data}"
            )


def main():
    rclpy.init()
    node = TracerFusionPolicyMpcRefNode()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
