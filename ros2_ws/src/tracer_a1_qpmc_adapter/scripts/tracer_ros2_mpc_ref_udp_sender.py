#!/usr/bin/env python3

import socket
import struct

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class TracerMpcRefUdpSender(Node):
    def __init__(self):
        super().__init__("tracer_ros2_mpc_ref_udp_sender")

        self.declare_parameter("udp_ip", "127.0.0.1")
        self.declare_parameter("udp_port", 50110)
        self.declare_parameter("topic", "/tracer/mpc_reference")

        self.udp_ip = self.get_parameter("udp_ip").value
        self.udp_port = int(self.get_parameter("udp_port").value)
        self.topic = self.get_parameter("topic").value

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.packet_fmt = "<6d"

        self.sub = self.create_subscription(
            Float64MultiArray,
            self.topic,
            self.callback,
            10,
        )

        self.get_logger().info(
            f"ROS2 {self.topic} -> UDP {self.udp_ip}:{self.udp_port}"
        )

    def callback(self, msg: Float64MultiArray):
        data = list(msg.data)

        if len(data) < 6:
            self.get_logger().warn(
                f"Expected >=6 values [counter,vx,yaw_rate,body_height,clearance,enable], got {len(data)}"
            )
            return

        values = [float(x) for x in data[:6]]
        packet = struct.pack(self.packet_fmt, *values)
        self.sock.sendto(packet, (self.udp_ip, self.udp_port))

        self.get_logger().info(
            f"sent counter={values[0]:.0f} vx={values[1]:.3f} yaw={values[2]:.3f} "
            f"h={values[3]:.3f} clr={values[4]:.3f} enable={values[5]:.1f}",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = TracerMpcRefUdpSender()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
