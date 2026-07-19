#!/usr/bin/env python3
import csv
import os
import sys
import time
from pathlib import Path

# Allow system /usr/bin/python3 ROS2 node to import torch from tracer_train env.
torch_site = os.environ.get("TRACER_PHASE_D6_TORCH_SITE_PACKAGES", "")
if torch_site and torch_site not in sys.path:
    sys.path.insert(0, torch_site)

import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy._rclpy_pybind11 import RCLError
from std_msgs.msg import Float64MultiArray, String


FEATURES = [
    "ctx_flat",
    "ctx_start_flat",
    "ctx_upslope",
    "ctx_rough",
    "ctx_downslope",
    "ctx_goal_flat",
    "ctx_unknown",
    "x",
    "y",
    "beta_motion",
    "beta_stability",
    "beta_energy",
    "ram_slip_proxy",
    "ram_roughness_proxy",
    "ram_sigma",
]

TARGETS = [
    "target_vx",
    "target_yaw_rate",
    "target_body_h",
    "target_clearance",
    "target_enable",
]


class MLPSelector(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def ref_from_msg(data):
    d = list(data)
    if len(d) < 6:
        return None
    return {
        "seq": float(d[0]),
        "vx": float(d[1]),
        "yaw_rate": float(d[2]),
        "body_h": float(d[3]),
        "clearance": float(d[4]),
        "enable": float(d[5]),
    }


class D6MLPSelectorShadow(Node):
    def __init__(self):
        super().__init__("tracer_phase_d6_mlp_selector_shadow_v0")

        self.model_pt = Path(os.environ.get(
            "TRACER_PHASE_D6_MLP_MODEL_PT",
            "models/phase_d6/d6_mlp_selector_from_d5_shadow_dataset_20260718_215059_v0.pt",
        ))
        if not self.model_pt.is_file():
            raise FileNotFoundError(f"MLP model not found: {self.model_pt}")

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = Path(os.environ.get(
            "TRACER_PHASE_D6_MLP_LOG",
            f"logs/phase_d6_mlp_shadow_{ts}/mlp_selector_shadow_v0.csv",
        ))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.context_topic = os.environ.get("TRACER_CONTEXT_LABEL_TOPIC", "/tracer/terrain_context_label")
        self.odom_topic = os.environ.get("TRACER_ODOM_TOPIC", "/tracer/robot_odom_flat")
        self.beta_topic = os.environ.get("TRACER_PHASE_D6_BETA_TOPIC", "/tracer/objective_beta")
        self.ram_topic = os.environ.get("TRACER_PHASE_D6_RAM_TOPIC", "/tracer/ram_mismatch")
        self.actual_ref_topic = os.environ.get("TRACER_PHASE_D6_MLP_ACTUAL_REF_TOPIC", "/tracer/empirical_mpc_reference")
        self.shadow_ref_topic = os.environ.get("TRACER_PHASE_D6_MLP_SHADOW_TOPIC", "/tracer/mlp_selector_shadow_ref")
        self.pub_hz = float(os.environ.get("TRACER_PHASE_D6_MLP_PUB_HZ", "10.0"))

        # Runtime safety clamps. The offline MLP is an unconstrained regressor,
        # so clamp outputs before publishing/logging. These bounds are chosen
        # around the validated D5 reference range.
        self.vx_min = float(os.environ.get("TRACER_PHASE_D6_MLP_VX_MIN", "0.000"))
        self.vx_max = float(os.environ.get("TRACER_PHASE_D6_MLP_VX_MAX", "0.240"))
        self.yaw_min = float(os.environ.get("TRACER_PHASE_D6_MLP_YAW_MIN", "-0.200"))
        self.yaw_max = float(os.environ.get("TRACER_PHASE_D6_MLP_YAW_MAX", "0.200"))
        self.body_h_min = float(os.environ.get("TRACER_PHASE_D6_MLP_BODY_H_MIN", "0.300"))
        self.body_h_max = float(os.environ.get("TRACER_PHASE_D6_MLP_BODY_H_MAX", "0.340"))
        self.clearance_min = float(os.environ.get("TRACER_PHASE_D6_MLP_CLEARANCE_MIN", "0.035"))
        self.clearance_max = float(os.environ.get("TRACER_PHASE_D6_MLP_CLEARANCE_MAX", "0.065"))
        self.enable_min = float(os.environ.get("TRACER_PHASE_D6_MLP_ENABLE_MIN", "0.000"))
        self.enable_max = float(os.environ.get("TRACER_PHASE_D6_MLP_ENABLE_MAX", "1.000"))

        ckpt = torch.load(self.model_pt, map_location="cpu")
        self.features = list(ckpt.get("features", FEATURES))
        self.targets = list(ckpt.get("targets", TARGETS))
        self.hidden_sizes = list(ckpt["hidden_sizes"])

        self.model = MLPSelector(len(self.features), len(self.targets), self.hidden_sizes)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.x_mean = ckpt["x_mean"].float()
        self.x_std = ckpt["x_std"].float()
        self.y_mean = ckpt["y_mean"].float()
        self.y_std = ckpt["y_std"].float()

        self.current_context = "unknown"
        self.x = 0.0
        self.y = 0.0
        self.beta = [0.33, 0.34, 0.33]
        self.ram = [0.10, 0.10, 0.20]
        self.actual_ref = None
        self.seq = 0

        self.pub = self.create_publisher(Float64MultiArray, self.shadow_ref_topic, 10)

        self.create_subscription(String, self.context_topic, self.on_context, 10)
        self.create_subscription(Float64MultiArray, self.odom_topic, self.on_odom, 10)
        self.create_subscription(Float64MultiArray, self.beta_topic, self.on_beta, 10)
        self.create_subscription(Float64MultiArray, self.ram_topic, self.on_ram, 10)
        self.create_subscription(Float64MultiArray, self.actual_ref_topic, self.on_actual_ref, 10)

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
            "actual_seq",
            "pred_seq",
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

        self.get_logger().info(f"torch_version={torch.__version__}")
        self.get_logger().info(f"model_pt={self.model_pt}")
        self.get_logger().info(f"context_topic={self.context_topic}")
        self.get_logger().info(f"odom_topic={self.odom_topic}")
        self.get_logger().info(f"beta_topic={self.beta_topic}")
        self.get_logger().info(f"ram_topic={self.ram_topic}")
        self.get_logger().info(f"actual_ref_topic={self.actual_ref_topic}")
        self.get_logger().info(f"shadow_ref_topic={self.shadow_ref_topic}")
        self.get_logger().info(f"log_path={self.log_path}")
        self.get_logger().info(
            "clamps: "
            f"vx=[{self.vx_min},{self.vx_max}], "
            f"yaw=[{self.yaw_min},{self.yaw_max}], "
            f"body_h=[{self.body_h_min},{self.body_h_max}], "
            f"clearance=[{self.clearance_min},{self.clearance_max}], "
            f"enable=[{self.enable_min},{self.enable_max}]"
        )

    def on_context(self, msg):
        self.current_context = msg.data.strip() or "unknown"

    def on_odom(self, msg):
        d = list(msg.data)
        if len(d) >= 2:
            self.x = float(d[0])
            self.y = float(d[1])

    def on_beta(self, msg):
        d = list(msg.data)
        if len(d) >= 3:
            self.beta = [float(d[0]), float(d[1]), float(d[2])]

    def on_ram(self, msg):
        d = list(msg.data)
        if len(d) >= 3:
            self.ram = [float(d[0]), float(d[1]), float(d[2])]

    def on_actual_ref(self, msg):
        self.actual_ref = ref_from_msg(msg.data)

    def context_onehot(self):
        labels = [
            "flat",
            "start_flat",
            "upslope",
            "rough",
            "downslope",
            "goal_flat",
            "unknown",
        ]
        ctx = self.current_context
        if ctx not in labels:
            ctx = "unknown"
        return [1.0 if ctx == label else 0.0 for label in labels]

    def build_feature_vector(self):
        return (
            self.context_onehot()
            + [float(self.x), float(self.y)]
            + [float(v) for v in self.beta]
            + [float(v) for v in self.ram]
        )

    @torch.no_grad()
    def predict(self):
        x = torch.tensor([self.build_feature_vector()], dtype=torch.float32)
        xn = (x - self.x_mean) / self.x_std
        yn = self.model(xn)
        y = yn * self.y_std + self.y_mean
        vals = y[0].detach().cpu().tolist()

        def clamp(v, lo, hi):
            return max(lo, min(hi, float(v)))

        return {
            "vx": clamp(vals[0], self.vx_min, self.vx_max),
            "yaw_rate": clamp(vals[1], self.yaw_min, self.yaw_max),
            "body_h": clamp(vals[2], self.body_h_min, self.body_h_max),
            "clearance": clamp(vals[3], self.clearance_min, self.clearance_max),
            "enable": clamp(vals[4], self.enable_min, self.enable_max),
        }

    def on_timer(self):
        self.seq += 1
        pred = self.predict()

        msg = Float64MultiArray()
        msg.data = [
            float(self.seq),
            pred["vx"],
            pred["yaw_rate"],
            pred["body_h"],
            pred["clearance"],
            pred["enable"],
        ]
        self.pub.publish(msg)

        a = self.actual_ref or {
            "seq": -1.0,
            "vx": 0.0,
            "yaw_rate": 0.0,
            "body_h": 0.32,
            "clearance": 0.045,
            "enable": 0.0,
        }

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
            "actual_seq": a["seq"],
            "pred_seq": self.seq,
            "actual_vx": a["vx"],
            "pred_vx": pred["vx"],
            "err_vx": pred["vx"] - a["vx"],
            "actual_yaw_rate": a["yaw_rate"],
            "pred_yaw_rate": pred["yaw_rate"],
            "err_yaw_rate": pred["yaw_rate"] - a["yaw_rate"],
            "actual_body_h": a["body_h"],
            "pred_body_h": pred["body_h"],
            "err_body_h": pred["body_h"] - a["body_h"],
            "actual_clearance": a["clearance"],
            "pred_clearance": pred["clearance"],
            "err_clearance": pred["clearance"] - a["clearance"],
            "actual_enable": a["enable"],
            "pred_enable": pred["enable"],
            "err_enable": pred["enable"] - a["enable"],
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
    node = D6MLPSelectorShadow()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
