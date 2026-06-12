#!/usr/bin/env python3
from __future__ import annotations

import json
import socket
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


ROBOT_STATE_LEN = 42
LOW_LEVEL_CMD_LEN = 61


class TracerRos2UdpBridge(Node):
    """Bridge between ROS2 topics and Isaac Lab conda process via UDP.

    Direction:
      Isaac Lab UDP -> /tracer/robot_state
      /tracer/low_level_cmd -> Isaac Lab UDP

    This avoids importing rclpy inside the Isaac Lab conda environment.
    """

    def __init__(self):
        super().__init__("tracer_ros2_udp_bridge")

        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("isaac_state_port", 50100)
        self.declare_parameter("isaac_cmd_port", 50101)
        self.declare_parameter("tick_hz", 100.0)
        self.declare_parameter("cmd_repeat_hz", 50.0)

        self.host = str(self.get_parameter("host").value)
        self.isaac_state_port = int(self.get_parameter("isaac_state_port").value)
        self.isaac_cmd_port = int(self.get_parameter("isaac_cmd_port").value)
        self.tick_hz = float(self.get_parameter("tick_hz").value)
        self.cmd_repeat_hz = float(self.get_parameter("cmd_repeat_hz").value)

        # Receive robot state from Isaac.
        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.state_sock.bind((self.host, self.isaac_state_port))
        self.state_sock.setblocking(False)

        # Send low-level command to Isaac.
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_addr = (self.host, self.isaac_cmd_port)

        self.robot_state_pub = self.create_publisher(
            Float64MultiArray,
            "/tracer/robot_state",
            10,
        )

        self.low_level_sub = self.create_subscription(
            Float64MultiArray,
            "/tracer/low_level_cmd",
            self.on_low_level_cmd,
            10,
        )

        self.latest_cmd: Optional[list[float]] = None
        self.latest_cmd_stamp = 0.0
        self.last_cmd_send_stamp = 0.0

        self.timer = self.create_timer(1.0 / max(self.tick_hz, 1.0), self.on_timer)

        self.get_logger().info(
            f"TracerRos2UdpBridge started. "
            f"UDP recv Isaac state: {self.host}:{self.isaac_state_port} -> /tracer/robot_state. "
            f"/tracer/low_level_cmd -> UDP send Isaac cmd: {self.host}:{self.isaac_cmd_port}."
        )

    def on_low_level_cmd(self, msg: Float64MultiArray):
        data = list(msg.data)
        if len(data) < LOW_LEVEL_CMD_LEN:
            self.get_logger().warn(
                f"Ignore short /tracer/low_level_cmd: len={len(data)}, expected>={LOW_LEVEL_CMD_LEN}",
                throttle_duration_sec=1.0,
            )
            return
        self.latest_cmd = data[:LOW_LEVEL_CMD_LEN]
        self.latest_cmd_stamp = time.time()
        self.send_latest_cmd()

    def send_latest_cmd(self):
        if self.latest_cmd is None:
            return
        payload = {
            "stamp": time.time(),
            "data": self.latest_cmd,
        }
        raw = json.dumps(payload).encode("utf-8")
        self.cmd_sock.sendto(raw, self.cmd_addr)
        self.last_cmd_send_stamp = time.time()

    def recv_isaac_states(self):
        while True:
            try:
                raw, _ = self.state_sock.recvfrom(65535)
            except BlockingIOError:
                break

            try:
                payload = json.loads(raw.decode("utf-8"))
                data = list(payload.get("data", []))
            except Exception as exc:
                self.get_logger().warn(f"Bad UDP robot_state packet: {repr(exc)}", throttle_duration_sec=1.0)
                continue

            if len(data) < ROBOT_STATE_LEN:
                self.get_logger().warn(
                    f"Ignore short UDP robot_state: len={len(data)}, expected>={ROBOT_STATE_LEN}",
                    throttle_duration_sec=1.0,
                )
                continue

            msg = Float64MultiArray()
            msg.data = data[:ROBOT_STATE_LEN]
            self.robot_state_pub.publish(msg)

    def on_timer(self):
        self.recv_isaac_states()

        # Repeat command at fixed rate so Isaac can receive it even if UDP drops one packet.
        now = time.time()
        if self.latest_cmd is not None:
            period = 1.0 / max(self.cmd_repeat_hz, 1.0)
            if now - self.last_cmd_send_stamp >= period:
                self.send_latest_cmd()


def main():
    rclpy.init()
    node = TracerRos2UdpBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
