#!/usr/bin/env python3

import socket
import struct
import time
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class TracerUdpOdomToRos2Node(Node):
    """
    UDP odom receiver.

    Input UDP packet:
      [stamp_wall, x_world, y_world, yaw_world, vx_world, vy_world]

    Output:
      /tracer/robot_odom_flat : Float64MultiArray
    """

    def __init__(self):
        super().__init__("tracer_udp_odom_to_ros2_node")

        self.declare_parameter("udp_ip", "0.0.0.0")
        self.declare_parameter("udp_port", 50120)
        self.declare_parameter("topic", "/tracer/robot_odom_flat")
        self.declare_parameter("publish_timeout_sec", 1.0)

        self.udp_ip = self.get_parameter("udp_ip").value
        self.udp_port = int(self.get_parameter("udp_port").value)
        self.topic = self.get_parameter("topic").value

        self.packet_fmt = "<6d"
        self.packet_size = struct.calcsize(self.packet_fmt)

        self.pub = self.create_publisher(Float64MultiArray, self.topic, 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))
        self.sock.setblocking(False)

        self.timer = self.create_timer(0.005, self.poll_udp)

        self.last_log_wall = 0.0

        # The ROS1 UDP payload includes a wall-clock stamp followed by world-frame
        # odometry. Downstream TRACER modules expect odom features, not timestamps.
        # Keep a local origin and publish relative odom:
        #   [x_rel, y_rel, z_rel, yaw_rel, vx_world, vy_world]
        self.origin_set = False
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_yaw = 0.0

        self.get_logger().info(
            f"UDP odom receiver listening on {self.udp_ip}:{self.udp_port} -> {self.topic}"
        )

    def poll_udp(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
            except BlockingIOError:
                return

            if len(data) < self.packet_size:
                self.get_logger().warn(f"short UDP odom packet: {len(data)} bytes")
                continue

            values = struct.unpack(self.packet_fmt, data[:self.packet_size])

            stamp_wall = float(values[0])
            x_world = float(values[1])
            y_world = float(values[2])
            yaw_world = float(values[3])
            vx_world = float(values[4])
            vy_world = float(values[5])

            if not self.origin_set:
                self.origin_x = x_world
                self.origin_y = y_world
                self.origin_yaw = yaw_world
                self.origin_set = True
                self.get_logger().info(
                    "set odom relative origin: "
                    f"stamp={stamp_wall:.3f} x0={self.origin_x:.3f} "
                    f"y0={self.origin_y:.3f} yaw0={self.origin_yaw:.3f}"
                )

            # Reject obviously invalid world-frame odom. The current ROS1 bridge can
            # occasionally send timestamp-like or uninitialized world values. Do not
            # feed those into RAM/metrics as clipped 100m signals.
            invalid_world = (
                abs(x_world) > 1.0e5
                or abs(y_world) > 1.0e5
                or abs(vx_world) > 20.0
                or abs(vy_world) > 20.0
            )

            if invalid_world:
                x_rel = 0.0
                y_rel = 0.0
                yaw_rel = 0.0
                vx_world = 0.0
                vy_world = 0.0
            else:
                x_rel = x_world - self.origin_x
                y_rel = y_world - self.origin_y
                yaw_rel = math.atan2(
                    math.sin(yaw_world - self.origin_yaw),
                    math.cos(yaw_world - self.origin_yaw),
                )

            # Output layout:
            #   [x_rel, y_rel, z_rel, yaw_rel, vx_world, vy_world]
            # z_rel is 0.0 because the current UDP payload does not include z.
            out = [x_rel, y_rel, 0.0, yaw_rel, vx_world, vy_world]

            # Final defensive clipping for downstream RAM/recorder safety.
            out = [max(min(float(v), 20.0), -20.0) for v in out]

            msg = Float64MultiArray()
            msg.data = out
            self.pub.publish(msg)

            now = time.time()
            if now - self.last_log_wall > 1.0:
                self.get_logger().info(
                    f"UDP->ROS2 odom from {addr[0]}:{addr[1]} "
                    f"x_rel={out[0]:.3f} y_rel={out[1]:.3f} yaw_rel={out[3]:.3f} "
                    f"vx={out[4]:.3f} vy={out[5]:.3f}"
                )
                self.last_log_wall = now


def main():
    rclpy.init()
    node = TracerUdpOdomToRos2Node()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
