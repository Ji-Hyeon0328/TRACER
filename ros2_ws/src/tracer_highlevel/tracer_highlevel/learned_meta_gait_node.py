#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import socket
import time
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class LearnedMetaGaitNode(Node):
    """
    M6.3 learned high-level command producer.

    Torch inference runs in the separate meta-gait UDP server.
    This ROS2 node remains torch-free and tracer_core-free.

    ROS output layout:
      [
        vx,
        yaw_rate,
        body_height,
        swing_clearance,
        gait_period,
        duty_factor,
      ]

    M6.3 default execution policy:
      - continuous fields come from the learned policy
      - structural fields use the characterized nominal reference

    Raw learned structural passthrough can be enabled explicitly later.
    """

    def __init__(self):
        super().__init__("tracer_learned_meta_gait_node")

        self.declare_parameter(
            "command_topic",
            "/tracer/meta_gait_cmd",
        )
        self.declare_parameter(
            "publish_hz",
            10.0,
        )

        self.declare_parameter(
            "policy_host",
            "127.0.0.1",
        )
        self.declare_parameter(
            "policy_port",
            50310,
        )
        self.declare_parameter(
            "policy_timeout_sec",
            0.05,
        )

        self.declare_parameter(
            "gait_mode",
            "fast",
        )

        self.declare_parameter(
            "beta_motion",
            1.0 / 3.0,
        )
        self.declare_parameter(
            "beta_stability",
            1.0 / 3.0,
        )
        self.declare_parameter(
            "beta_energy",
            1.0 / 3.0,
        )

        self.declare_parameter(
            "ram_level",
            "unknown",
        )
        self.declare_parameter(
            "ram_sigma",
            0.0,
        )
        self.declare_parameter(
            "ram_control_risk",
            0.0,
        )
        self.declare_parameter(
            "ram_fallen_prob",
            0.0,
        )
        self.declare_parameter(
            "ram_recovery_prob",
            0.0,
        )

        self.declare_parameter(
            "use_learned_structural",
            False,
        )
        self.declare_parameter(
            "nominal_gait_period",
            1.0 / 1.4,
        )
        self.declare_parameter(
            "nominal_duty_factor",
            0.65,
        )

        self.command_topic = str(
            self.get_parameter("command_topic").value
        )
        self.publish_hz = float(
            self.get_parameter("publish_hz").value
        )

        self.policy_host = str(
            self.get_parameter("policy_host").value
        )
        self.policy_port = int(
            self.get_parameter("policy_port").value
        )
        self.policy_timeout_sec = float(
            self.get_parameter("policy_timeout_sec").value
        )

        if (
            not math.isfinite(self.publish_hz)
            or self.publish_hz <= 0.0
        ):
            raise ValueError(
                "publish_hz must be finite and > 0"
            )

        if (
            not math.isfinite(self.policy_timeout_sec)
            or self.policy_timeout_sec <= 0.0
        ):
            raise ValueError(
                "policy_timeout_sec must be finite and > 0"
            )

        self.publisher = self.create_publisher(
            Float64MultiArray,
            self.command_topic,
            10,
        )

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )
        self.sock.settimeout(
            self.policy_timeout_sec
        )

        self.sequence = 0
        self.failure_count = 0
        self.last_signature = None

        self.timer = self.create_timer(
            1.0 / self.publish_hz,
            self.publish_command,
        )

        self.get_logger().info(
            "TRACER M6.3 learned high-level node started: "
            f"{self.command_topic} @ {self.publish_hz:.1f} Hz; "
            f"policy={self.policy_host}:{self.policy_port}; "
            f"mode={self.get_parameter('gait_mode').value}; "
            "default structural execution="
            "characterized nominal"
        )

    def p_float(self, name: str) -> float:
        value = float(
            self.get_parameter(name).value
        )

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        return value

    def build_request(self) -> dict[str, Any]:
        self.sequence += 1

        return {
            "request_id": (
                f"m6_3_ros_{self.sequence}_"
                f"{time.time():.6f}"
            ),
            "gait_mode": str(
                self.get_parameter(
                    "gait_mode"
                ).value
            ),
            "context": [],
            "robot_state": [],
            "goal": [],
            "beta": {
                "motion": self.p_float(
                    "beta_motion"
                ),
                "stability": self.p_float(
                    "beta_stability"
                ),
                "energy": self.p_float(
                    "beta_energy"
                ),
            },
            "ram": {
                "rho": [],
                "sigma": self.p_float(
                    "ram_sigma"
                ),
                "ram_level": str(
                    self.get_parameter(
                        "ram_level"
                    ).value
                ),
                "control_risk": self.p_float(
                    "ram_control_risk"
                ),
                "fallen_prob": self.p_float(
                    "ram_fallen_prob"
                ),
                "recovery_prob": self.p_float(
                    "ram_recovery_prob"
                ),
            },
        }

    def query_policy(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        packet = json.dumps(
            request
        ).encode("utf-8")

        self.sock.sendto(
            packet,
            (
                self.policy_host,
                self.policy_port,
            ),
        )

        data, _addr = self.sock.recvfrom(
            65535
        )

        response = json.loads(
            data.decode("utf-8")
        )

        if not isinstance(response, dict):
            raise RuntimeError(
                "policy response must be a JSON object"
            )

        if response.get("request_id") != request["request_id"]:
            raise RuntimeError(
                "policy response request_id mismatch"
            )

        if not response.get("ok", False):
            raise RuntimeError(
                "policy server error: "
                f"{response.get('error_type', 'Error')}: "
                f"{response.get('error', '<unknown>')}"
            )

        meta = response.get(
            "meta_gait",
            None,
        )

        if not isinstance(meta, dict):
            raise RuntimeError(
                "policy response missing meta_gait"
            )

        return meta

    @staticmethod
    def finite_meta(
        meta: dict[str, Any],
        name: str,
    ) -> float:
        value = float(meta[name])

        if not math.isfinite(value):
            raise ValueError(
                f"learned {name} must be finite"
            )

        return value

    def command_values(
        self,
        meta: dict[str, Any],
    ) -> list[float]:
        vx = self.finite_meta(
            meta,
            "vx",
        )
        yaw_rate = self.finite_meta(
            meta,
            "yaw_rate",
        )
        body_height = self.finite_meta(
            meta,
            "body_height",
        )
        swing_clearance = self.finite_meta(
            meta,
            "swing_clearance",
        )

        learned_period = self.finite_meta(
            meta,
            "gait_period",
        )
        learned_duty = self.finite_meta(
            meta,
            "duty_factor",
        )

        use_learned_structural = bool(
            self.get_parameter(
                "use_learned_structural"
            ).value
        )

        if use_learned_structural:
            gait_period = learned_period
            duty_factor = learned_duty
            structural_source = "learned"
        else:
            gait_period = self.p_float(
                "nominal_gait_period"
            )
            duty_factor = self.p_float(
                "nominal_duty_factor"
            )
            structural_source = "nominal"

        if gait_period <= 0.0:
            raise ValueError(
                "gait_period must be > 0"
            )

        signature = (
            round(vx, 4),
            round(yaw_rate, 4),
            round(body_height, 4),
            round(swing_clearance, 4),
            round(learned_period, 4),
            round(learned_duty, 4),
            structural_source,
        )

        if signature != self.last_signature:
            self.last_signature = signature

            self.get_logger().info(
                "M6.3 learned command: "
                f"vx={vx:.4f} "
                f"yaw={yaw_rate:.4f} "
                f"h={body_height:.4f} "
                f"clr={swing_clearance:.4f} "
                f"learned_T={learned_period:.4f} "
                f"learned_D={learned_duty:.4f} "
                f"applied_request_T={gait_period:.4f} "
                f"applied_request_D={duty_factor:.4f} "
                f"structural={structural_source}"
            )

        return [
            vx,
            yaw_rate,
            body_height,
            swing_clearance,
            gait_period,
            duty_factor,
        ]

    def publish_command(self):
        try:
            request = self.build_request()
            meta = self.query_policy(
                request
            )
            values = self.command_values(
                meta
            )

        except Exception as exc:
            self.failure_count += 1

            if (
                self.failure_count == 1
                or self.failure_count % 20 == 0
            ):
                self.get_logger().warning(
                    "M6.3 learned inference unavailable; "
                    "not publishing command: "
                    f"{type(exc).__name__}: {exc}"
                )

            # Important:
            # do not synthesize a replacement command here.
            # M5 source freshness / transport fallback remains
            # the authoritative downstream behavior.
            return

        self.failure_count = 0

        msg = Float64MultiArray()
        msg.data = values
        self.publisher.publish(msg)

    def destroy_node(self):
        try:
            self.sock.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LearnedMetaGaitNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
