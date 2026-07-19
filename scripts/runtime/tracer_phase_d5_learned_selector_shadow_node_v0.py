#!/usr/bin/env python3
import csv
import json
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float64MultiArray, String


CONTEXTS = ["flat", "start_flat", "upslope", "rough", "downslope", "goal_flat", "unknown"]


def fnum(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


class LearnedSelectorShadow(Node):
    def __init__(self):
        super().__init__("tracer_phase_d5_learned_selector_shadow_v0")

        self.model_json = Path(os.environ.get(
            "TRACER_PHASE_D5_LEARNED_MODEL_JSON",
            "models/phase_d5/d5_learned_selector_ridge_from_shadow_dataset_20260718_215059_v0.json",
        ))

        if not self.model_json.is_file():
            raise FileNotFoundError(f"learned selector model not found: {self.model_json}")

        self.model = json.loads(self.model_json.read_text())
        self.features = list(self.model["features"])
        self.targets = list(self.model["targets"])
        self.models = self.model["models"]

        ts = time.strftime("%Y%m%d_%H%M%S")
        default_log = f"logs/phase_d5_learned_shadow_{ts}/learned_selector_shadow_v0.csv"
        self.log_path = Path(os.environ.get("TRACER_PHASE_D5_LEARNED_LOG", default_log))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.context_topic = os.environ.get("TRACER_CONTEXT_LABEL_TOPIC", "/tracer/terrain_context_label")
        self.odom_topic = os.environ.get("TRACER_ODOM_TOPIC", "/tracer/robot_odom_flat")
        self.beta_topic = os.environ.get("TRACER_PHASE_D5_BETA_TOPIC", "/tracer/objective_beta")
        self.ram_topic = os.environ.get("TRACER_PHASE_D5_RAM_TOPIC", "/tracer/ram_mismatch")
        self.actual_ref_topic = os.environ.get("TRACER_REF_TOPIC", "/tracer/mpc_reference")
        self.shadow_ref_topic = os.environ.get("TRACER_PHASE_D5_LEARNED_SHADOW_TOPIC", "/tracer/learned_selector_shadow_ref")
        self.pub_hz = float(os.environ.get("TRACER_PHASE_D5_LEARNED_PUB_HZ", "10.0"))

        self.current_context = "unknown"
        self.x = 0.0
        self.y = 0.0
        self.beta = [0.33, 0.34, 0.33]
        self.ram = [0.0, 0.0, 0.0]
        self.actual_ref = [None, None, None, None, None, None]
        self.seq = 0

        self.create_subscription(String, self.context_topic, self.on_context, 10)
        self.create_subscription(Float64MultiArray, self.odom_topic, self.on_odom, 10)
        self.create_subscription(Float64MultiArray, self.beta_topic, self.on_beta, 10)
        self.create_subscription(Float64MultiArray, self.ram_topic, self.on_ram, 10)
        self.create_subscription(Float64MultiArray, self.actual_ref_topic, self.on_actual_ref, 10)

        self.pub = self.create_publisher(Float64MultiArray, self.shadow_ref_topic, 10)

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
            "pred_seq",
            "actual_seq",
            "actual_vx",
            "pred_vx",
            "err_vx",
            "actual_yaw_rate",
            "pred_yaw_rate",
            "err_yaw_rate",
            "actual_body_h",
            "pred_body_h",
            "err_body_h",
            "actual_clearance",
            "pred_clearance",
            "err_clearance",
            "actual_enable",
            "pred_enable",
            "err_enable",
        ])
        self.writer.writeheader()

        self.timer = self.create_timer(1.0 / self.pub_hz, self.on_timer)

        self.get_logger().info(f"model_json={self.model_json}")
        self.get_logger().info(f"shadow_ref_topic={self.shadow_ref_topic}")
        self.get_logger().info(f"actual_ref_topic={self.actual_ref_topic}")
        self.get_logger().info(f"log_path={self.log_path}")

    def on_context(self, msg):
        label = msg.data.strip()
        self.current_context = label if label else "unknown"

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

    def on_actual_ref(self, msg):
        data = list(msg.data)
        for i in range(min(6, len(data))):
            self.actual_ref[i] = float(data[i])

    def feature_dict(self):
        ctx = self.current_context if self.current_context in CONTEXTS else "unknown"
        d = {f"ctx_{c}": 1.0 if c == ctx else 0.0 for c in CONTEXTS}
        d.update({
            "x": float(self.x),
            "y": float(self.y),
            "beta_motion": float(self.beta[0]),
            "beta_stability": float(self.beta[1]),
            "beta_energy": float(self.beta[2]),
            "ram_slip_proxy": float(self.ram[0]),
            "ram_roughness_proxy": float(self.ram[1]),
            "ram_sigma": float(self.ram[2]),
        })
        return d

    def predict_target(self, target, fd):
        m = self.models[target]
        x = [float(fd.get(name, 0.0)) for name in self.features]
        return sum(a * b for a, b in zip(x, m["weights"])) + float(m["bias"])

    @staticmethod
    def err(pred, actual):
        if actual is None:
            return ""
        return pred - actual

    def on_timer(self):
        self.seq += 1
        fd = self.feature_dict()

        pred = {target: self.predict_target(target, fd) for target in self.targets}

        pred_vx = pred.get("target_vx", 0.0)
        pred_yaw = pred.get("target_yaw_rate", 0.0)
        pred_h = pred.get("target_body_h", 0.32)
        pred_clr = pred.get("target_clearance", 0.045)
        pred_enable = pred.get("target_enable", 1.0)

        msg = Float64MultiArray()
        msg.data = [
            float(self.seq),
            float(pred_vx),
            float(pred_yaw),
            float(pred_h),
            float(pred_clr),
            float(pred_enable),
        ]
        self.pub.publish(msg)

        aseq, avx, ayaw, ah, aclr, aenable = self.actual_ref

        self.writer.writerow({
            "t_wall": time.time(),
            "context": self.current_context,
            "x": self.x,
            "y": self.y,
            "beta_motion": self.beta[0],
            "beta_stability": self.beta[1],
            "beta_energy": self.beta[2],
            "ram_slip_proxy": self.ram[0],
            "ram_roughness_proxy": self.ram[1],
            "ram_sigma": self.ram[2],
            "pred_seq": self.seq,
            "actual_seq": "" if aseq is None else aseq,
            "actual_vx": "" if avx is None else avx,
            "pred_vx": pred_vx,
            "err_vx": self.err(pred_vx, avx),
            "actual_yaw_rate": "" if ayaw is None else ayaw,
            "pred_yaw_rate": pred_yaw,
            "err_yaw_rate": self.err(pred_yaw, ayaw),
            "actual_body_h": "" if ah is None else ah,
            "pred_body_h": pred_h,
            "err_body_h": self.err(pred_h, ah),
            "actual_clearance": "" if aclr is None else aclr,
            "pred_clearance": pred_clr,
            "err_clearance": self.err(pred_clr, aclr),
            "actual_enable": "" if aenable is None else aenable,
            "pred_enable": pred_enable,
            "err_enable": self.err(pred_enable, aenable),
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
    node = LearnedSelectorShadow()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
