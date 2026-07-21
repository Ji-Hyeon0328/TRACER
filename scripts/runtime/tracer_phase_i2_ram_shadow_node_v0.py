#!/usr/bin/env python3
import csv
import json
import math
import os
import time
from collections import Counter, deque
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


def env_float(name, default):
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def fmean(vals):
    vals = [float(v) for v in vals if finite(v)]
    return sum(vals) / len(vals) if vals else None


def fstd(vals):
    vals = [float(v) for v in vals if finite(v)]
    if len(vals) <= 1:
        return 0.0 if vals else None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def context_key(label):
    label = (label or "unknown").strip()
    known = {
        "flat",
        "start_flat",
        "rough",
        "upslope",
        "downslope",
        "goal_flat",
        "unknown",
    }
    return label if label in known else "unknown"


def thirds(vals):
    vals = list(vals)
    n = len(vals)
    if n == 0:
        return [], [], []
    if n < 3:
        return vals, vals, vals
    a = max(1, n // 3)
    b = max(a + 1, 2 * n // 3)
    return vals[:a], vals[a:b], vals[b:]


class I2RAMShadow(Node):
    def __init__(self):
        super().__init__("tracer_phase_i2_ram_shadow_node_v0")

        self.model_path = Path(os.environ.get(
            "TRACER_PHASE_I2_RAM_MODEL_JSON",
            "models/phase_i/i1_ram_ridge_predictor_v0.json",
        ))

        self.context_topic = os.environ.get("TRACER_PHASE_I2_CONTEXT_TOPIC", "/tracer/terrain_context_label")
        self.odom_topic = os.environ.get("TRACER_PHASE_I2_ODOM_TOPIC", "/tracer/robot_odom_flat")
        self.ref_topic = os.environ.get("TRACER_PHASE_I2_REF_TOPIC", "/tracer/empirical_mpc_reference")
        self.legacy_ram_topic = os.environ.get("TRACER_PHASE_I2_LEGACY_RAM_TOPIC", "/tracer/ram_mismatch")
        self.out_topic = os.environ.get("TRACER_PHASE_I2_RAM_OUT_TOPIC", "/tracer/ram_mismatch_i2_shadow")

        self.window_s = env_float("TRACER_PHASE_I2_WINDOW_S", 8.0)
        self.pub_hz = env_float("TRACER_PHASE_I2_PUB_HZ", 10.0)
        self.default_context = os.environ.get("TRACER_PHASE_I2_DEFAULT_CONTEXT", "unknown")

        self.model = self._load_model()
        self.feature_cols = self.model["feature_cols"]
        self.feature_means = [float(x) for x in self.model["feature_means"]]
        self.feature_stds = [float(x) if abs(float(x)) > 1e-12 else 1.0 for x in self.model["feature_stds"]]
        self.W = self.model["coef_with_intercept"]

        if len(self.W) != len(self.feature_cols) + 1:
            raise RuntimeError("I2 RAM model coefficient shape mismatch")

        self.ctx = self.default_context
        self.x = None
        self.y = None
        self.ref = [None, None, None, None, None]  # vx, yaw, body_h, clearance, enable
        self.legacy_ram = [None, None, None]       # slip, roughness, sigma

        self.samples = deque()

        self.pub = self.create_publisher(Float64MultiArray, self.out_topic, 10)

        self.create_subscription(String, self.context_topic, self.on_context, 10)
        self.create_subscription(Float64MultiArray, self.odom_topic, self.on_odom, 10)
        self.create_subscription(Float64MultiArray, self.ref_topic, self.on_ref, 10)
        self.create_subscription(Float64MultiArray, self.legacy_ram_topic, self.on_legacy_ram, 10)

        log_dir = Path(os.environ.get("TRACER_PHASE_I2_LOG_DIR", "logs/phase_i/i2_ram_shadow_latest"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = log_dir / "ram_shadow_i2_v0.csv"
        self.log_f = open(self.log_csv, "w", newline="")
        self.writer = csv.DictWriter(
            self.log_f,
            fieldnames=[
                "t_wall",
                "context",
                "x",
                "y",
                "window_n",
                "observed_features",
                "imputed_features",
                "rho_slip_pred",
                "rho_rough_pred",
                "sigma_pred",
                "raw_slip",
                "raw_rough",
                "raw_sigma",
                "legacy_slip",
                "legacy_rough",
                "legacy_sigma",
                "legacy_l1",
                "ref_vx",
                "ref_yaw_rate",
                "ref_clearance",
            ],
        )
        self.writer.writeheader()
        self.log_f.flush()

        dt = 1.0 / max(self.pub_hz, 1e-6)
        self.timer = self.create_timer(dt, self.on_timer)

        self.get_logger().info(f"[TRACER:I2] loaded RAM model={self.model_path}")
        self.get_logger().info(f"[TRACER:I2] publishing learned RAM shadow to {self.out_topic}")
        self.get_logger().info(f"[TRACER:I2] comparing legacy RAM from {self.legacy_ram_topic}")
        self.get_logger().info(f"[TRACER:I2] logging to {self.log_csv}")

    def _load_model(self):
        if not self.model_path.exists():
            raise RuntimeError(f"missing I2 RAM model JSON: {self.model_path}")
        root = json.loads(self.model_path.read_text())
        return root["model"]

    def on_context(self, msg):
        self.ctx = context_key(msg.data)

    def on_odom(self, msg):
        data = list(msg.data)
        if len(data) >= 2:
            self.x = float(data[0])
            self.y = float(data[1])

    def on_ref(self, msg):
        data = list(msg.data)
        if len(data) >= 2:
            self.ref[0] = float(data[1])
        if len(data) >= 3:
            self.ref[1] = float(data[2])
        if len(data) >= 4:
            self.ref[2] = float(data[3])
        if len(data) >= 5:
            self.ref[3] = float(data[4])
        if len(data) >= 6:
            self.ref[4] = float(data[5])

    def on_legacy_ram(self, msg):
        data = list(msg.data)
        if len(data) >= 1:
            self.legacy_ram[0] = float(data[0])
        if len(data) >= 2:
            self.legacy_ram[1] = float(data[1])
        if len(data) >= 3:
            self.legacy_ram[2] = float(data[2])

    def add_stats(self, feat, prefix, vals):
        m = fmean(vals)
        s = fstd(vals)
        if m is not None:
            feat[f"{prefix}_mean_g1"] = m
        if s is not None:
            feat[f"{prefix}_std_g1"] = s

    def snapshot(self, now):
        self.samples.append({
            "t": now,
            "ctx": context_key(self.ctx),
            "x": self.x,
            "y": self.y,
            "ref": list(self.ref),
        })

        while self.samples and now - self.samples[0]["t"] > self.window_s:
            self.samples.popleft()

    def build_features(self):
        feat = {}
        samples = list(self.samples)
        if not samples:
            return feat

        ctxs = [context_key(s["ctx"]) for s in samples]
        cnt = Counter(ctxs)
        n = float(len(ctxs))
        for k in ["flat", "start_flat", "rough", "upslope", "downslope", "goal_flat", "unknown"]:
            feat[f"context_frac_{k}"] = cnt.get(k, 0) / n
        feat["context_unique_count"] = float(len(cnt))

        ref_vx = [s["ref"][0] for s in samples if finite(s["ref"][0])]
        ref_yaw = [s["ref"][1] for s in samples if finite(s["ref"][1])]
        ref_clear = [s["ref"][3] for s in samples if finite(s["ref"][3])]
        self.add_stats(feat, "ref_vx_mean", ref_vx)
        self.add_stats(feat, "ref_yaw_rate_mean", ref_yaw)
        self.add_stats(feat, "ref_clearance_mean", ref_clear)

        odom = [s for s in samples if finite(s["x"]) and finite(s["y"])]
        if len(odom) >= 1:
            xs = [float(s["x"]) for s in odom]
            ys = [float(s["y"]) for s in odom]

            feat["x_progress_g1"] = xs[-1] - xs[0]
            feat["y_change_g1"] = ys[-1] - ys[0]
            feat["abs_y_change_g1"] = abs(ys[-1] - ys[0])

            ye, _, yl = thirds(ys)
            aye, _, ayl = thirds([abs(y) for y in ys])

            if ye and yl:
                feat["y_late_minus_early_g1"] = fmean(yl) - fmean(ye)
                feat["abs_y_late_minus_early_g1"] = fmean(ayl) - fmean(aye)

            dx = xs[-1] - xs[0]
            dy = ys[-1] - ys[0]
            feat["lateral_per_forward_g1"] = abs(dy) / max(abs(dx), 0.05)

            if len(odom) >= 2:
                dt = odom[-1]["t"] - odom[0]["t"]
                if dt > 1e-6:
                    vx_est = dx / dt
                    rvx = fmean(ref_vx)
                    if rvx is not None:
                        feat["vx_tracking_error_g1"] = vx_est - rvx
                        feat["vx_tracking_abs_error_g1"] = abs(vx_est - rvx)
                        feat["vx_tracking_ratio_g1"] = vx_est / rvx if abs(rvx) > 1e-4 else 0.0

        return feat

    def predict_ram(self, feat):
        z = []
        observed = 0
        imputed = 0

        for i, col in enumerate(self.feature_cols):
            val = feat.get(col, None)
            if finite(val):
                x = float(val)
                observed += 1
            else:
                x = self.feature_means[i]
                imputed += 1

            z.append((x - self.feature_means[i]) / self.feature_stds[i])

        raw = []
        for j in range(3):
            y = float(self.W[0][j])
            for i, zi in enumerate(z):
                y += zi * float(self.W[i + 1][j])
            raw.append(y)

        pred = [clamp01(x) for x in raw]
        return raw, pred, observed, imputed

    def on_timer(self):
        now = time.time()
        self.snapshot(now)
        feat = self.build_features()
        raw, pred, observed, imputed = self.predict_ram(feat)

        msg = Float64MultiArray()
        msg.data = [float(pred[0]), float(pred[1]), float(pred[2])]
        self.pub.publish(msg)

        legacy_l1 = ""
        if all(finite(x) for x in self.legacy_ram):
            legacy_l1 = sum(abs(pred[i] - self.legacy_ram[i]) for i in range(3))

        self.writer.writerow({
            "t_wall": f"{now:.6f}",
            "context": context_key(self.ctx),
            "x": "" if self.x is None else f"{self.x:.6f}",
            "y": "" if self.y is None else f"{self.y:.6f}",
            "window_n": len(self.samples),
            "observed_features": observed,
            "imputed_features": imputed,
            "rho_slip_pred": f"{pred[0]:.8f}",
            "rho_rough_pred": f"{pred[1]:.8f}",
            "sigma_pred": f"{pred[2]:.8f}",
            "raw_slip": f"{raw[0]:.8f}",
            "raw_rough": f"{raw[1]:.8f}",
            "raw_sigma": f"{raw[2]:.8f}",
            "legacy_slip": "" if self.legacy_ram[0] is None else f"{self.legacy_ram[0]:.6f}",
            "legacy_rough": "" if self.legacy_ram[1] is None else f"{self.legacy_ram[1]:.6f}",
            "legacy_sigma": "" if self.legacy_ram[2] is None else f"{self.legacy_ram[2]:.6f}",
            "legacy_l1": "" if legacy_l1 == "" else f"{legacy_l1:.8f}",
            "ref_vx": "" if self.ref[0] is None else f"{self.ref[0]:.6f}",
            "ref_yaw_rate": "" if self.ref[1] is None else f"{self.ref[1]:.6f}",
            "ref_clearance": "" if self.ref[3] is None else f"{self.ref[3]:.6f}",
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
    node = I2RAMShadow()
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
