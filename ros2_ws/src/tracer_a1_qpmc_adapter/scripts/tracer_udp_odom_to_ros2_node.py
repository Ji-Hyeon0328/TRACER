#!/usr/bin/env python3

import socket
import struct
import time

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

            msg = Float64MultiArray()
            msg.data = list(values)
            self.pub.publish(msg)

            now = time.time()
            if now - self.last_log_wall > 1.0:
                self.get_logger().info(
                    f"UDP->ROS2 odom from {addr[0]}:{addr[1]} "
                    f"x={values[1]:.3f} y={values[2]:.3f} yaw={values[3]:.3f} "
                    f"vx={values[4]:.3f} vy={values[5]:.3f}"
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
