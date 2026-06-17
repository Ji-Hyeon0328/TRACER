#!/usr/bin/env python3

import socket
import struct
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class TracerUdpProprioToRos2Node(Node):
    """
    UDP proprio receiver.

    Input UDP packet:
      45 doubles

    Output:
      /tracer/proprio_vector Float64MultiArray

    Layout:
      [0] stamp_wall

      [1:4]   base position x,y,z
      [4:7]   roll,pitch,yaw
      [7:10]  base linear velocity vx,vy,vz
      [10:13] base angular velocity wx,wy,wz

      [13:25] joint_pos[12]
      [25:37] joint_vel[12]

      [37:41] contact_binary[4]     FL,FR,RL,RR
      [41:45] contact_force_z[4]    FL,FR,RL,RR
    """

    def __init__(self):
        super().__init__("tracer_udp_proprio_to_ros2_node")

        self.declare_parameter("udp_ip", "0.0.0.0")
        self.declare_parameter("udp_port", 50130)
        self.declare_parameter("topic", "/tracer/proprio_vector")

        self.udp_ip = self.get_parameter("udp_ip").value
        self.udp_port = int(self.get_parameter("udp_port").value)
        self.topic = self.get_parameter("topic").value

        self.packet_fmt = "<45d"
        self.packet_size = struct.calcsize(self.packet_fmt)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))
        self.sock.setblocking(False)

        self.pub = self.create_publisher(Float64MultiArray, self.topic, 10)
        self.timer = self.create_timer(0.002, self.poll_udp)

        self.last_log_wall = 0.0

        self.get_logger().info(
            f"UDP proprio receiver listening on {self.udp_ip}:{self.udp_port} -> {self.topic}"
        )

    def poll_udp(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                return

            if len(data) < self.packet_size:
                self.get_logger().warn(f"short UDP proprio packet: {len(data)} bytes")
                continue

            values = struct.unpack(self.packet_fmt, data[:self.packet_size])

            msg = Float64MultiArray()
            msg.data = list(values)
            self.pub.publish(msg)

            now = time.time()
            if now - self.last_log_wall > 1.0:
                contacts = [int(x) for x in values[37:41]]
                self.get_logger().info(
                    f"UDP->ROS2 proprio from {addr[0]}:{addr[1]} "
                    f"pos=({values[1]:.2f},{values[2]:.2f},{values[3]:.2f}) "
                    f"rpy=({values[4]:.2f},{values[5]:.2f},{values[6]:.2f}) "
                    f"v=({values[7]:.2f},{values[8]:.2f},{values[9]:.2f}) "
                    f"contacts={contacts}"
                )
                self.last_log_wall = now


def main():
    rclpy.init()
    node = TracerUdpProprioToRos2Node()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
