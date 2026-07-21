#!/usr/bin/env python3
import csv
import json
import math
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))


def env_float(name, default):
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def context_key(label):
    label = (label or "unknown").strip()
    known = {"flat", "start_flat", "rough", "upslope", "downslope", "goal_flat", "unknown"}
    return label if label in known else "unknown"


class I5SafeRAMPostprocess(Node):
    def __init__(self):
        super().__init__("tracer_phase_i5_safe_ram_postprocess_node_v0")

        self.postprocess_json = Path(os.environ.get(
            "TRACER_PHASE_I5_POSTPROCESS_JSON",
            "models/phase_i/i4_damped_ram_postprocess_v0.json",
        ))

        self.learned_topic = os.environ.get("TRACER_PHASE_I5_LEARNED_RAM_TOPIC", "/tracer/ram_mismatch_i2_shadow")
        self.legacy_topic = os.environ.get("TRACER_PHASE_I5_LEGACY_RAM_TOPIC", "/tracer/ram_mismatch")
        self.out_topic = os.environ.get("TRACER_PHASE_I5_RAM_OUT_TOPIC", "/tracer/ram_mismatch_i5_safe_shadow")
        self.context_topic = os.environ.get("TRACER_PHASE_I5_CONTEXT_TOPIC", "/tracer/terrain_context_label")
        self.odom_topic = os.environ.get("TRACER_PHASE_I5_ODOM_TOPIC", "/tracer/robot_odom_flat")

        self.pub_hz = env_float("TRACER_PHASE_I5_PUB_HZ", 10.0)
        self.max_age_s = env_float("TRACER_PHASE_I5_MAX_AGE_S", 1.5)

        self.cfg = self.load_cfg()
        prior = self.cfg["prior"]
        pp = self.cfg["postprocess"]

        self.prior = [
            float(prior["rho_slip"]),
            float(prior["rho_rough"]),
            float(prior["sigma"]),
        ]
        self.gain = float(pp["gain"])
        self.legacy_blend = float(pp["legacy_blend"])
        self.floor = float(pp["floor"])
        self.ceil = float(pp["ceil"])

        self.learned = None
        self.learned_t = None
        self.legacy = None
        self.legacy_t = None
        self.context = "unknown"
        self.x = None
        self.y = None

        self.pub = self.create_publisher(Float64MultiArray, self.out_topic, 10)

        self.create_subscription(Float64MultiArray, self.learned_topic, self.on_learned, 10)
        self.create_subscription(Float64MultiArray, self.legacy_topic, self.on_legacy, 10)
        self.create_subscription(String, self.context_topic, self.on_context, 10)
        self.create_subscription(Float64MultiArray, self.odom_topic, self.on_odom, 10)

        log_dir = Path(os.environ.get("TRACER_PHASE_I5_LOG_DIR", "logs/phase_i/i5_safe_ram_shadow_latest"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = log_dir / "safe_ram_shadow_i5_v0.csv"
        self.log_f = open(self.log_csv, "w", newline="")
        self.writer = csv.DictWriter(
            self.log_f,
            fieldnames=[
                "t_wall",
                "context",
                "x",
                "y",
                "used_legacy",
                "learned_age",
                "legacy_age",
                "learned_slip",
                "learned_rough",
                "learned_sigma",
                "damped_slip",
                "damped_rough",
                "damped_sigma",
                "safe_slip",
                "safe_rough",
                "safe_sigma",
                "legacy_slip",
                "legacy_rough",
                "legacy_sigma",
                "learned_l1_legacy",
                "damped_l1_legacy",
                "safe_l1_legacy",
            ],
        )
        self.writer.writeheader()
        self.log_f.flush()

        self.timer = self.create_timer(1.0 / max(self.pub_hz, 1e-6), self.on_timer)

        self.get_logger().info(f"[TRACER:I5] postprocess={self.postprocess_json}")
        self.get_logger().info(f"[TRACER:I5] learned RAM input={self.learned_topic}")
        self.get_logger().info(f"[TRACER:I5] legacy RAM input={self.legacy_topic}")
        self.get_logger().info(f"[TRACER:I5] safe RAM output={self.out_topic}")
        self.get_logger().info(f"[TRACER:I5] log={self.log_csv}")

    def load_cfg(self):
        if not self.postprocess_json.exists():
            raise RuntimeError(f"missing postprocess json: {self.postprocess_json}")
        return json.loads(self.postprocess_json.read_text())

    def on_context(self, msg):
        self.context = context_key(msg.data)

    def on_odom(self, msg):
        data = list(msg.data)
        if len(data) >= 2:
            self.x = float(data[0])
            self.y = float(data[1])

    def on_learned(self, msg):
        data = list(msg.data)
        if len(data) >= 3:
            self.learned = [float(data[0]), float(data[1]), float(data[2])]
            self.learned_t = time.time()

    def on_legacy(self, msg):
        data = list(msg.data)
        if len(data) >= 3:
            self.legacy = [float(data[0]), float(data[1]), float(data[2])]
            self.legacy_t = time.time()

    def l1(self, a, b):
        if a is None or b is None:
            return None
        if not all(finite(v) for v in a + b):
            return None
        return sum(abs(a[i] - b[i]) for i in range(3))

    def postprocess(self, learned, legacy, use_legacy):
        damped = [
            clamp(self.prior[i] + self.gain * (learned[i] - self.prior[i]), self.floor, self.ceil)
            for i in range(3)
        ]

        if use_legacy:
            safe = [
                clamp((1.0 - self.legacy_blend) * legacy[i] + self.legacy_blend * damped[i], self.floor, self.ceil)
                for i in range(3)
            ]
        else:
            safe = damped[:]

        return damped, safe

    def on_timer(self):
        now = time.time()

        if self.learned is None or self.learned_t is None:
            return

        learned_age = now - self.learned_t
        if learned_age > self.max_age_s:
            return

        use_legacy = False
        legacy_age = None
        if self.legacy is not None and self.legacy_t is not None:
            legacy_age = now - self.legacy_t
            use_legacy = legacy_age <= self.max_age_s

        legacy = self.legacy if use_legacy else None
        damped, safe = self.postprocess(self.learned, legacy, use_legacy)

        msg = Float64MultiArray()
        msg.data = [float(safe[0]), float(safe[1]), float(safe[2])]
        self.pub.publish(msg)

        learned_l1 = self.l1(self.learned, legacy)
        damped_l1 = self.l1(damped, legacy)
        safe_l1 = self.l1(safe, legacy)

        self.writer.writerow({
            "t_wall": f"{now:.6f}",
            "context": context_key(self.context),
            "x": "" if self.x is None else f"{self.x:.6f}",
            "y": "" if self.y is None else f"{self.y:.6f}",
            "used_legacy": "1" if use_legacy else "0",
            "learned_age": f"{learned_age:.6f}",
            "legacy_age": "" if legacy_age is None else f"{legacy_age:.6f}",

            "learned_slip": f"{self.learned[0]:.8f}",
            "learned_rough": f"{self.learned[1]:.8f}",
            "learned_sigma": f"{self.learned[2]:.8f}",

            "damped_slip": f"{damped[0]:.8f}",
            "damped_rough": f"{damped[1]:.8f}",
            "damped_sigma": f"{damped[2]:.8f}",

            "safe_slip": f"{safe[0]:.8f}",
            "safe_rough": f"{safe[1]:.8f}",
            "safe_sigma": f"{safe[2]:.8f}",

            "legacy_slip": "" if legacy is None else f"{legacy[0]:.8f}",
            "legacy_rough": "" if legacy is None else f"{legacy[1]:.8f}",
            "legacy_sigma": "" if legacy is None else f"{legacy[2]:.8f}",

            "learned_l1_legacy": "" if learned_l1 is None else f"{learned_l1:.8f}",
            "damped_l1_legacy": "" if damped_l1 is None else f"{damped_l1:.8f}",
            "safe_l1_legacy": "" if safe_l1 is None else f"{safe_l1:.8f}",
        })
        self.log_f.flush()

    def destroy_node(self):
        try:
            self.log_f.flush()
            self.log_f.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = I5SafeRAMPostprocess()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
