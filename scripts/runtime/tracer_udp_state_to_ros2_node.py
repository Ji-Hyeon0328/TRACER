#!/usr/bin/env python3
import json
import socket
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


class TracerUdpStateToRos2(Node):
    def __init__(self):
        super().__init__("tracer_udp_state_to_ros2_node")

        self.bind_ip = self.declare_parameter("bind_ip", "0.0.0.0").value
        self.bind_port = int(self.declare_parameter("bind_port", 50130).value)
        self.publish_hz = float(self.declare_parameter("publish_hz", 50.0).value)

        self.pub = self.create_publisher(Float64MultiArray, "/tracer/robot_state_flat", 10)
        self.meta_pub = self.create_publisher(String, "/tracer/robot_state_meta", 10)

        self.latest = None
        self.latest_time = 0.0
        self.lock = threading.Lock()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.bind_ip, self.bind_port))
        self.sock.settimeout(0.2)

        self.thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.thread.start()

        self.timer = self.create_timer(1.0 / self.publish_hz, self.publish_latest)

        self.get_logger().info(
            f"TRACER UDP state receiver listening on {self.bind_ip}:{self.bind_port}"
        )

    def recv_loop(self):
        while rclpy.ok():
            try:
                data, _addr = self.sock.recvfrom(65535)
                obj = json.loads(data.decode("utf-8"))
                with self.lock:
                    self.latest = obj
                    self.latest_time = time.time()
            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().warn(f"UDP state decode error: {e}")

    def publish_latest(self):
        with self.lock:
            obj = self.latest

        if not obj:
            return

        q = [float(x) for x in obj.get("q", [0.0] * 12)]
        dq = [float(x) for x in obj.get("dq", [0.0] * 12)]
        contacts = [float(x) for x in obj.get("contacts", [0.0] * 4)]
        forces = [float(x) for x in obj.get("contact_forces", [0.0] * 4)]

        # Flat layout:
        # 0 seq
        # 1 age_sec
        # 2 base_z
        # 3 base_roll
        # 4 base_pitch
        # 5 base_yaw
        # 6..17 q[12]
        # 18..29 dq[12]
        # 30..33 contacts[4]
        # 34..37 contact_forces[4]
        stamp = float(obj.get("stamp", time.time()))
        age = max(0.0, time.time() - stamp)

        arr = [
            float(obj.get("seq", 0.0)),
            age,
            float(obj.get("base_z", 0.0)),
            float(obj.get("base_roll", 0.0)),
            float(obj.get("base_pitch", 0.0)),
            float(obj.get("base_yaw", 0.0)),
        ] + q[:12] + dq[:12] + contacts[:4] + forces[:4]

        msg = Float64MultiArray()
        msg.data = arr
        self.pub.publish(msg)

        meta = {
            "schema": "tracer_robot_state_flat_meta_v0",
            "layout": [
                "seq",
                "age_sec",
                "base_z",
                "base_roll",
                "base_pitch",
                "base_yaw",
                "q_FL_hip", "q_FL_thigh", "q_FL_calf",
                "q_FR_hip", "q_FR_thigh", "q_FR_calf",
                "q_RL_hip", "q_RL_thigh", "q_RL_calf",
                "q_RR_hip", "q_RR_thigh", "q_RR_calf",
                "dq_FL_hip", "dq_FL_thigh", "dq_FL_calf",
                "dq_FR_hip", "dq_FR_thigh", "dq_FR_calf",
                "dq_RL_hip", "dq_RL_thigh", "dq_RL_calf",
                "dq_RR_hip", "dq_RR_thigh", "dq_RR_calf",
                "contact_FL", "contact_FR", "contact_RL", "contact_RR",
                "force_FL", "force_FR", "force_RL", "force_RR",
            ],
            "joint_order": obj.get("joint_order", []),
            "contact_order": obj.get("contact_order", []),
            "raw_joint_names": obj.get("raw_joint_names", []),
        }
        m = String()
        m.data = json.dumps(meta, sort_keys=True)
        self.meta_pub.publish(m)


def main():
    rclpy.init()
    node = TracerUdpStateToRos2()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
