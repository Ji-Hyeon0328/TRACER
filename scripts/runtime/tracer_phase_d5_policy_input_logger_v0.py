#!/usr/bin/env python3
import csv
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray


class D5PolicyInputLogger(Node):
    def __init__(self):
        super().__init__("tracer_phase_d5_policy_input_logger_v0")

        ts = time.strftime("%Y%m%d_%H%M%S")
        default_log = f"logs/phase_d5_shadow_inputs_{ts}.csv"
        self.log_path = Path(os.environ.get("TRACER_PHASE_D5_INPUT_LOG", default_log))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.context_topic = os.environ.get("TRACER_CONTEXT_LABEL_TOPIC", "/tracer/terrain_context_label")
        self.odom_topic = os.environ.get("TRACER_ODOM_TOPIC", "/tracer/robot_odom_flat")
        self.beta_topic = os.environ.get("TRACER_PHASE_D5_BETA_TOPIC", "/tracer/objective_beta")
        self.ram_topic = os.environ.get("TRACER_PHASE_D5_RAM_TOPIC", "/tracer/ram_mismatch")
        self.ref_topic = os.environ.get("TRACER_REF_TOPIC", "/tracer/mpc_reference")
        self.log_hz = float(os.environ.get("TRACER_PHASE_D5_LOG_HZ", "10.0"))

        self.label = "unknown"
        self.x = None
        self.y = None
        self.beta = [None, None, None]
        self.ram = [None, None, None]
        self.ref = [None, None, None, None, None, None]

        self.create_subscription(String, self.context_topic, self.on_context, 10)
        self.create_subscription(Float64MultiArray, self.odom_topic, self.on_odom, 10)
        self.create_subscription(Float64MultiArray, self.beta_topic, self.on_beta, 10)
        self.create_subscription(Float64MultiArray, self.ram_topic, self.on_ram, 10)
        self.create_subscription(Float64MultiArray, self.ref_topic, self.on_ref, 10)

        self.f = self.log_path.open("w", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=[
            "t_wall",
            "context",
            "x",
            "y",
            "beta_motion",
            "beta_stability",
            "beta_energy",
            "ram_slip_proxy",
            "ram_roughness_proxy",
            "ram_sigma",
            "ref_seq",
            "ref_vx",
            "ref_yaw_rate",
            "ref_body_h",
            "ref_clearance",
            "ref_enable",
        ])
        self.writer.writeheader()

        self.timer = self.create_timer(1.0 / self.log_hz, self.on_timer)

        self.get_logger().info(f"writing D5 policy input log: {self.log_path}")
        self.get_logger().info(f"context_topic={self.context_topic}")
        self.get_logger().info(f"odom_topic={self.odom_topic}")
        self.get_logger().info(f"beta_topic={self.beta_topic}")
        self.get_logger().info(f"ram_topic={self.ram_topic}")
        self.get_logger().info(f"ref_topic={self.ref_topic}")

    def on_context(self, msg):
        self.label = msg.data.strip() or "unknown"

    def on_odom(self, msg):
        data = list(msg.data)
        if len(data) >= 2:
            self.x = float(data[0])
            self.y = float(data[1])

    def on_beta(self, msg):
        data = list(msg.data)
        for i in range(min(3, len(data))):
            self.beta[i] = float(data[i])

    def on_ram(self, msg):
        data = list(msg.data)
        for i in range(min(3, len(data))):
            self.ram[i] = float(data[i])

    def on_ref(self, msg):
        data = list(msg.data)
        for i in range(min(6, len(data))):
            self.ref[i] = float(data[i])

    def on_timer(self):
        self.writer.writerow({
            "t_wall": time.time(),
            "context": self.label,
            "x": self.x,
            "y": self.y,
            "beta_motion": self.beta[0],
            "beta_stability": self.beta[1],
            "beta_energy": self.beta[2],
            "ram_slip_proxy": self.ram[0],
            "ram_roughness_proxy": self.ram[1],
            "ram_sigma": self.ram[2],
            "ref_seq": self.ref[0],
            "ref_vx": self.ref[1],
            "ref_yaw_rate": self.ref[2],
            "ref_body_h": self.ref[3],
            "ref_clearance": self.ref[4],
            "ref_enable": self.ref[5],
        })
        self.f.flush()

    def destroy_node(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = D5PolicyInputLogger()
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
