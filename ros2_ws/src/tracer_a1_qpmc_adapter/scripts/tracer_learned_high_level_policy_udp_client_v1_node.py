#!/usr/bin/env python3

import socket
import struct
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def normalize_beta(beta):
    beta = [max(0.0, float(x)) for x in beta[:3]]
    s = sum(beta)
    if s <= 1e-9:
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    return [x / s for x in beta]


class TracerLearnedHighLevelPolicyUdpClientV1Node(Node):
    """
    Learned high-level policy v1 ROS2 wrapper.

    Inputs:
      /tracer/relative_goal    [4]
      /tracer/context_vector   [16]
      /tracer/latent_vector    [16]
      /tracer/objective_weights [3]

    UDP request:
      39 doubles

    UDP response:
      [vx, yaw_rate, body_height, clearance, enable]

    Output:
      /tracer/mpc_reference
        [counter, vx, yaw_rate, body_height, clearance, enable]
    """

    def __init__(self):
        super().__init__("tracer_learned_high_level_policy_udp_client_v1_node")

        self.declare_parameter("relative_goal_topic", "/tracer/relative_goal")
        self.declare_parameter("context_topic", "/tracer/context_vector")
        self.declare_parameter("latent_topic", "/tracer/latent_vector")
        self.declare_parameter("objective_topic", "/tracer/objective_weights")
        self.declare_parameter("mpc_ref_topic", "/tracer/mpc_reference")

        self.declare_parameter("server_ip", "127.0.0.1")
        self.declare_parameter("server_port", 50211)
        self.declare_parameter("local_ip", "0.0.0.0")
        self.declare_parameter("local_port", 0)

        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("input_timeout_sec", 1.0)
        self.declare_parameter("server_timeout_sec", 0.5)

        self.relative_goal_topic = self.get_parameter("relative_goal_topic").value
        self.context_topic = self.get_parameter("context_topic").value
        self.latent_topic = self.get_parameter("latent_topic").value
        self.objective_topic = self.get_parameter("objective_topic").value
        self.mpc_ref_topic = self.get_parameter("mpc_ref_topic").value

        self.server_ip = self.get_parameter("server_ip").value
        self.server_port = int(self.get_parameter("server_port").value)
        self.local_ip = self.get_parameter("local_ip").value
        self.local_port = int(self.get_parameter("local_port").value)

        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.input_timeout_sec = float(self.get_parameter("input_timeout_sec").value)
        self.server_timeout_sec = float(self.get_parameter("server_timeout_sec").value)

        self.input_fmt = "<39d"
        self.output_fmt = "<5d"
        self.output_size = struct.calcsize(self.output_fmt)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.local_ip, self.local_port))
        self.sock.setblocking(False)

        self.counter = 0.0

        self.last_goal: Optional[List[float]] = None
        self.last_goal_time = None

        self.last_context: Optional[List[float]] = None
        self.last_context_time = None

        self.last_latent: Optional[List[float]] = None
        self.last_latent_time = None

        self.last_beta: Optional[List[float]] = None
        self.last_beta_time = None

        self.last_send_wall = 0.0
        self.last_recv_wall = 0.0
        self.last_log_wall = 0.0

        self.pub = self.create_publisher(Float64MultiArray, self.mpc_ref_topic, 10)

        self.create_subscription(Float64MultiArray, self.relative_goal_topic, self.goal_callback, 10)
        self.create_subscription(Float64MultiArray, self.context_topic, self.context_callback, 10)
        self.create_subscription(Float64MultiArray, self.latent_topic, self.latent_callback, 10)
        self.create_subscription(Float64MultiArray, self.objective_topic, self.objective_callback, 10)

        self.timer = self.create_timer(1.0 / max(1.0, self.publish_hz), self.timer_callback)
        self.poll_timer = self.create_timer(0.002, self.poll_response)

        self.get_logger().info(
            f"learned HL policy v1 UDP client: "
            f"{self.relative_goal_topic} + {self.context_topic} + {self.latent_topic} + {self.objective_topic} "
            f"-> UDP {self.server_ip}:{self.server_port} -> {self.mpc_ref_topic}"
        )

    def goal_callback(self, msg):
        data = list(msg.data)
        if len(data) < 2:
            self.get_logger().warn(f"relative_goal expects at least [x_rel,y_rel], got {len(data)}")
            return

        out = [0.0, 0.0, 0.0, 1.0]
        for i in range(min(4, len(data))):
            out[i] = float(data[i])

        self.last_goal = out
        self.last_goal_time = self.get_clock().now()

    def context_callback(self, msg):
        data = list(msg.data)
        if len(data) < 16:
            self.get_logger().warn(f"context_vector expects >=16 values, got {len(data)}")
            return

        self.last_context = [float(x) for x in data[:16]]
        self.last_context_time = self.get_clock().now()

    def latent_callback(self, msg):
        data = list(msg.data)
        if len(data) < 16:
            self.get_logger().warn(f"latent_vector expects >=16 values, got {len(data)}")
            return

        self.last_latent = [float(x) for x in data[:16]]
        self.last_latent_time = self.get_clock().now()

    def objective_callback(self, msg):
        data = list(msg.data)
        if len(data) < 3:
            self.get_logger().warn(f"objective_weights expects 3 values, got {len(data)}")
            return

        self.last_beta = normalize_beta(data[:3])
        self.last_beta_time = self.get_clock().now()

    def age_sec(self, stamp):
        if stamp is None:
            return 1e9
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def publish_stop(self, reason: str):
        self.counter += 1.0
        msg = Float64MultiArray()
        msg.data = [
            self.counter,
            0.0,
            0.0,
            0.30,
            0.03,
            0.0,
        ]
        self.pub.publish(msg)
        self.get_logger().info(f"publish stop: {reason}", throttle_duration_sec=1.0)

    def timer_callback(self):
        if self.last_goal is None:
            self.publish_stop("waiting for relative_goal")
            return
        if self.last_context is None:
            self.publish_stop("waiting for context_vector")
            return
        if self.last_latent is None:
            self.publish_stop("waiting for latent_vector")
            return
        if self.last_beta is None:
            self.publish_stop("waiting for objective_weights")
            return

        goal_age = self.age_sec(self.last_goal_time)
        context_age = self.age_sec(self.last_context_time)
        latent_age = self.age_sec(self.last_latent_time)
        beta_age = self.age_sec(self.last_beta_time)

        if goal_age > self.input_timeout_sec:
            self.publish_stop(f"relative_goal timeout age={goal_age:.2f}s")
            return
        if context_age > self.input_timeout_sec:
            self.publish_stop(f"context timeout age={context_age:.2f}s")
            return
        if latent_age > self.input_timeout_sec:
            self.publish_stop(f"latent timeout age={latent_age:.2f}s")
            return
        if beta_age > self.input_timeout_sec:
            self.publish_stop(f"objective timeout age={beta_age:.2f}s")
            return

        request = self.last_goal + self.last_context + self.last_latent + self.last_beta

        if len(request) != 39:
            self.publish_stop(f"request length mismatch: {len(request)}")
            return

        packet = struct.pack(self.input_fmt, *request)
        self.sock.sendto(packet, (self.server_ip, self.server_port))
        self.last_send_wall = time.time()

        if self.last_recv_wall > 0.0:
            server_age = time.time() - self.last_recv_wall
            if server_age > self.server_timeout_sec:
                self.publish_stop(f"server response timeout age={server_age:.2f}s")

    def poll_response(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                return

            if len(data) < self.output_size:
                self.get_logger().warn(f"short learned HL v1 response: {len(data)} bytes")
                continue

            values = struct.unpack(self.output_fmt, data[:self.output_size])
            vx, yaw_rate, body_height, clearance, enable = values

            self.counter += 1.0

            msg = Float64MultiArray()
            msg.data = [
                self.counter,
                float(vx),
                float(yaw_rate),
                float(body_height),
                float(clearance),
                float(enable),
            ]
            self.pub.publish(msg)

            self.last_recv_wall = time.time()

            now = time.time()
            if now - self.last_log_wall > 1.0:
                beta = self.last_beta if self.last_beta is not None else [0.0, 0.0, 0.0]
                self.get_logger().info(
                    f"learned HL v1 response from {addr[0]}:{addr[1]} "
                    f"beta=({beta[0]:.2f},{beta[1]:.2f},{beta[2]:.2f}) "
                    f"vx={vx:.3f} yaw={yaw_rate:.3f} "
                    f"h={body_height:.3f} clr={clearance:.3f} enable={enable:.1f}"
                )
                self.last_log_wall = now


def main():
    rclpy.init()
    node = TracerLearnedHighLevelPolicyUdpClientV1Node()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
