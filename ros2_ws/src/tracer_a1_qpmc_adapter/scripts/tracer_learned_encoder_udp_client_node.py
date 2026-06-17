#!/usr/bin/env python3

import socket
import struct
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class TracerLearnedEncoderUdpClientNode(Node):
    """
    ROS2 wrapper for learned proprio encoder.

    Input:
      /tracer/proprio_vector, 45 doubles

    UDP request:
      45 doubles -> learned encoder server

    UDP response:
      32 doubles = context[16] + latent[16]

    Outputs:
      /tracer/context_vector
      /tracer/latent_vector

    This node does not import torch.
    Torch inference runs in env_isaaclab through the UDP server.
    """

    def __init__(self):
        super().__init__("tracer_learned_encoder_udp_client_node")

        self.declare_parameter("proprio_topic", "/tracer/proprio_vector")
        self.declare_parameter("context_topic", "/tracer/context_vector")
        self.declare_parameter("latent_topic", "/tracer/latent_vector")

        self.declare_parameter("server_ip", "127.0.0.1")
        self.declare_parameter("server_port", 50200)
        self.declare_parameter("local_ip", "0.0.0.0")
        self.declare_parameter("local_port", 0)

        self.proprio_topic = self.get_parameter("proprio_topic").value
        self.context_topic = self.get_parameter("context_topic").value
        self.latent_topic = self.get_parameter("latent_topic").value

        self.server_ip = self.get_parameter("server_ip").value
        self.server_port = int(self.get_parameter("server_port").value)
        self.local_ip = self.get_parameter("local_ip").value
        self.local_port = int(self.get_parameter("local_port").value)

        self.input_fmt = "<45d"
        self.output_fmt = "<32d"
        self.output_size = struct.calcsize(self.output_fmt)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.local_ip, self.local_port))
        self.sock.setblocking(False)

        self.pub_context = self.create_publisher(Float64MultiArray, self.context_topic, 10)
        self.pub_latent = self.create_publisher(Float64MultiArray, self.latent_topic, 10)

        self.create_subscription(
            Float64MultiArray,
            self.proprio_topic,
            self.proprio_callback,
            10,
        )

        self.timer = self.create_timer(0.002, self.poll_response)

        self.last_send_wall = 0.0
        self.last_recv_wall = 0.0
        self.last_log_wall = 0.0

        self.get_logger().info(
            f"learned encoder UDP client: {self.proprio_topic} "
            f"-> UDP {self.server_ip}:{self.server_port} "
            f"-> {self.context_topic}, {self.latent_topic}"
        )

    def proprio_callback(self, msg: Float64MultiArray):
        data = list(msg.data)
        if len(data) < 45:
            self.get_logger().warn(f"proprio_vector expects >=45 values, got {len(data)}")
            return

        values = [float(x) for x in data[:45]]
        packet = struct.pack(self.input_fmt, *values)
        self.sock.sendto(packet, (self.server_ip, self.server_port))
        self.last_send_wall = time.time()

    def poll_response(self):
        received_any = False

        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                break

            if len(data) < self.output_size:
                self.get_logger().warn(f"short learned encoder response: {len(data)} bytes")
                continue

            values = struct.unpack(self.output_fmt, data[:self.output_size])

            context = list(values[:16])
            latent = list(values[16:32])

            context_msg = Float64MultiArray()
            latent_msg = Float64MultiArray()

            context_msg.data = context
            latent_msg.data = latent

            self.pub_context.publish(context_msg)
            self.pub_latent.publish(latent_msg)

            self.last_recv_wall = time.time()
            received_any = True

            now = time.time()
            if now - self.last_log_wall > 1.0:
                self.get_logger().info(
                    f"learned encoder response from {addr[0]}:{addr[1]} "
                    f"context h={context[0]:.3f} roll={context[1]:.3f} pitch={context[2]:.3f} "
                    f"speed={context[6]:.3f} contact={context[8]:.3f} | "
                    f"latent risk={latent[0]:.3f} low_contact={latent[1]:.3f} slip={latent[2]:.3f}"
                )
                self.last_log_wall = now

        now = time.time()
        if self.last_send_wall > 0.0 and self.last_recv_wall < self.last_send_wall:
            if now - self.last_log_wall > 2.0:
                self.get_logger().warn(
                    "waiting for learned encoder UDP response. "
                    "Is tracer_proprio_encoder_udp_server_v0.py running?"
                )
                self.last_log_wall = now


def main():
    rclpy.init()
    node = TracerLearnedEncoderUdpClientNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
