#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import socket
import time
from typing import Optional


import rclpy
from rclpy.node import Node

from std_msgs.msg import (
    Bool,
    Float64MultiArray,
    String,
)


COMMAND_SCHEMA = (
    "tracer.meta_gait.command.v1"
)

TELEMETRY_SCHEMA = (
    "tracer.pympc.telemetry.v1"
)

COMMAND_LEN = 6


class MetaGaitUdpBridge(Node):
    """
    ROS2 sidecar for PyMPC.

    ROS2 -> UDP:
      /tracer/meta_gait_cmd
        [vx, yaw_rate, body_height,
         swing_clearance, gait_period,
         duty_factor]
      -> localhost UDP command packet

    UDP -> ROS2:
      PyMPC telemetry
      -> applied command / safety / override
    """

    def __init__(self):
        super().__init__(
            "tracer_pympc_meta_gait_udp_bridge"
        )

        self.declare_parameter(
            "udp_host",
            "127.0.0.1",
        )

        self.declare_parameter(
            "command_udp_port",
            50510,
        )

        self.declare_parameter(
            "telemetry_udp_port",
            50511,
        )

        self.declare_parameter(
            "command_topic",
            "/tracer/meta_gait_cmd",
        )

        self.declare_parameter(
            "command_repeat_hz",
            20.0,
        )

        self.declare_parameter(
            "command_source_timeout_s",
            0.25,
        )

        self.declare_parameter(
            "telemetry_poll_hz",
            100.0,
        )

        self.host = str(
            self.get_parameter(
                "udp_host"
            ).value
        )

        self.command_port = int(
            self.get_parameter(
                "command_udp_port"
            ).value
        )

        self.telemetry_port = int(
            self.get_parameter(
                "telemetry_udp_port"
            ).value
        )

        self.command_topic = str(
            self.get_parameter(
                "command_topic"
            ).value
        )

        self.command_repeat_hz = float(
            self.get_parameter(
                "command_repeat_hz"
            ).value
        )

        self.command_source_timeout_s = float(
            self.get_parameter(
                "command_source_timeout_s"
            ).value
        )

        self.telemetry_poll_hz = float(
            self.get_parameter(
                "telemetry_poll_hz"
            ).value
        )

        if self.command_source_timeout_s <= 0.0:
            raise ValueError(
                "command_source_timeout_s must be > 0"
            )

        self.command_addr = (
            self.host,
            self.command_port,
        )

        self.command_sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.telemetry_sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.telemetry_sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        self.telemetry_sock.bind(
            (
                self.host,
                self.telemetry_port,
            )
        )

        self.telemetry_sock.setblocking(
            False
        )

        self.command_sub = (
            self.create_subscription(
                Float64MultiArray,
                self.command_topic,
                self.on_command,
                10,
            )
        )

        self.applied_pub = (
            self.create_publisher(
                Float64MultiArray,
                "/tracer/pympc/applied_meta_gait",
                10,
            )
        )

        self.safety_pub = (
            self.create_publisher(
                String,
                "/tracer/pympc/safety_state",
                10,
            )
        )

        self.override_pub = (
            self.create_publisher(
                Bool,
                "/tracer/pympc/override_active",
                10,
            )
        )

        self.status_pub = (
            self.create_publisher(
                String,
                "/tracer/pympc/status",
                10,
            )
        )

        self.latest_values: (
            Optional[list[float]]
        ) = None

        self.latest_seq = 0

        self.last_command_receive: (
            Optional[float]
        ) = None

        self.source_stale_announced = False

        self.last_command_send = 0.0

        timer_hz = max(
            self.telemetry_poll_hz,
            self.command_repeat_hz,
            1.0,
        )

        self.timer = self.create_timer(
            1.0 / timer_hz,
            self.on_timer,
        )

        self.get_logger().info(
            "TRACER PyMPC ROS2 sidecar started: "
            f"{self.command_topic} "
            f"-> UDP {self.host}:{self.command_port}; "
            f"telemetry UDP "
            f"{self.host}:{self.telemetry_port} "
            "-> /tracer/pympc/*"
        )

    @staticmethod
    def validate_values(
        values,
    ) -> list[float]:
        if len(values) != COMMAND_LEN:
            raise ValueError(
                "expected exactly six values "
                "[vx,yaw_rate,body_height,"
                "swing_clearance,gait_period,"
                "duty_factor]"
            )

        result = [
            float(x)
            for x in values
        ]

        if not all(
            math.isfinite(x)
            for x in result
        ):
            raise ValueError(
                "all command values "
                "must be finite"
            )

        if result[4] <= 0.0:
            raise ValueError(
                "gait_period must be > 0"
            )

        return result

    def on_command(
        self,
        msg: Float64MultiArray,
    ):
        try:
            values = self.validate_values(
                list(msg.data)
            )
        except Exception as exc:
            self.get_logger().warn(
                "Ignore invalid "
                f"{self.command_topic}: "
                f"{exc}",
                throttle_duration_sec=1.0,
            )
            return

        self.latest_seq += 1
        self.latest_values = values

        self.last_command_receive = (
            time.monotonic()
        )

        self.source_stale_announced = False

        self.send_latest_command()

        self.get_logger().info(
            "accepted meta-gait "
            f"seq={self.latest_seq} "
            f"vx={values[0]:.3f} "
            f"yaw={values[1]:.3f} "
            f"h={values[2]:.3f} "
            f"clr={values[3]:.3f} "
            f"T={values[4]:.3f} "
            f"D={values[5]:.3f}"
        )

    def send_latest_command(self):
        if self.latest_values is None:
            return

        (
            vx,
            yaw_rate,
            body_height,
            swing_clearance,
            gait_period,
            duty_factor,
        ) = self.latest_values

        payload = {
            "schema": COMMAND_SCHEMA,
            "seq": int(self.latest_seq),
            "stamp": time.time(),
            "label": "ros2",
            "command": {
                "vx": vx,
                "yaw_rate": yaw_rate,
                "body_height":
                    body_height,
                "swing_clearance":
                    swing_clearance,
                "gait_period":
                    gait_period,
                "duty_factor":
                    duty_factor,
            },
        }

        raw = json.dumps(
            payload,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        self.command_sock.sendto(
            raw,
            self.command_addr,
        )

        self.last_command_send = (
            time.monotonic()
        )

    def recv_telemetry(self):
        while True:
            try:
                raw, _ = (
                    self.telemetry_sock
                    .recvfrom(65535)
                )

            except BlockingIOError:
                break

            try:
                payload = json.loads(
                    raw.decode("utf-8")
                )

                if (
                    payload.get("schema")
                    != TELEMETRY_SCHEMA
                ):
                    raise ValueError(
                        "unexpected schema"
                    )

            except Exception as exc:
                self.get_logger().warn(
                    "Bad PyMPC telemetry "
                    f"packet: {exc}",
                    throttle_duration_sec=1.0,
                )
                continue

            applied = payload.get(
                "applied"
            )

            if isinstance(applied, dict):
                try:
                    msg = Float64MultiArray()

                    msg.data = [
                        float(applied["vx"]),
                        float(
                            applied["yaw_rate"]
                        ),
                        float(
                            applied[
                                "body_height"
                            ]
                        ),
                        float(
                            applied[
                                "swing_clearance"
                            ]
                        ),
                        float(
                            applied[
                                "gait_period"
                            ]
                        ),
                        float(
                            applied[
                                "duty_factor"
                            ]
                        ),
                    ]

                    self.applied_pub.publish(
                        msg
                    )

                except Exception as exc:
                    self.get_logger().warn(
                        "Bad applied command "
                        f"in telemetry: {exc}",
                        throttle_duration_sec=1.0,
                    )

            safety_msg = String()
            safety_msg.data = str(
                payload.get(
                    "safety_state",
                    "unknown",
                )
            )

            self.safety_pub.publish(
                safety_msg
            )

            override_msg = Bool()
            override_msg.data = bool(
                payload.get(
                    "override_active",
                    False,
                )
            )

            self.override_pub.publish(
                override_msg
            )

            status_msg = String()

            status_msg.data = json.dumps(
                {
                    "telemetry_seq":
                        payload.get("seq"),
                    "command_seq":
                        payload.get(
                            "command_seq"
                        ),
                    "requested_label":
                        payload.get(
                            "requested_label"
                        ),
                    "selected_label":
                        payload.get(
                            "selected_label"
                        ),
                    "command_fresh":
                        payload.get(
                            "command_fresh"
                        ),
                    "command_age_s":
                        payload.get(
                            "command_age_s"
                        ),
                    "structural_commit":
                        payload.get(
                            "structural_commit"
                        ),
                    "sim_time_s":
                        payload.get(
                            "sim_time_s"
                        ),
                    "structural_commit_count":
                        payload.get(
                            "structural_commit_count"
                        ),
                    "override_reasons":
                        payload.get(
                            "override_reasons",
                            [],
                        ),
                },
                separators=(",", ":"),
            )

            self.status_pub.publish(
                status_msg
            )

    def on_timer(self):
        now = time.monotonic()

        if (
            self.latest_values is not None
            and self.last_command_receive
            is not None
        ):
            source_age = (
                now
                - self.last_command_receive
            )

            if (
                source_age
                <= self.command_source_timeout_s
            ):
                period = (
                    1.0
                    / max(
                        self.command_repeat_hz,
                        1.0,
                    )
                )

                if (
                    now
                    - self.last_command_send
                    >= period
                ):
                    self.send_latest_command()

            elif not self.source_stale_announced:
                self.get_logger().warn(
                    "Meta-gait source stale: "
                    f"age={source_age:.3f}s > "
                    f"{self.command_source_timeout_s:.3f}s; "
                    "stop UDP command forwarding"
                )

                self.source_stale_announced = True

        self.recv_telemetry()

    def destroy_node(self):
        try:
            self.command_sock.close()
            self.telemetry_sock.close()
        finally:
            return super().destroy_node()


def main():
    rclpy.init()

    node = MetaGaitUdpBridge()

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
