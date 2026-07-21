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


def env_int(name, default):
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        return int(raw)
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


def fmin(vals):
    vals = [float(v) for v in vals if finite(v)]
    return min(vals) if vals else None


def fmax(vals):
    vals = [float(v) for v in vals if finite(v)]
    return max(vals) if vals else None


def project_simplex(v):
    v = [float(x) if finite(x) else 1.0 / 3.0 for x in v]
    n = len(v)
    u = sorted(v, reverse=True)
    cssv = []
    s = 0.0
    for x in u:
        s += x
        cssv.append(s - 1.0)

    rho = -1
    for i, x in enumerate(u):
        if x - cssv[i] / float(i + 1) > 0:
            rho = i

    if rho < 0:
        return [1.0 / n] * n

    theta = cssv[rho] / float(rho + 1)
    w = [max(x - theta, 0.0) for x in v]
    sw = sum(w)
    if sw <= 1e-12:
        return [1.0 / n] * n
    return [x / sw for x in w]


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


class H2ObjectiveSelectorShadow(Node):
    def __init__(self):
        super().__init__("tracer_phase_h2_objective_selector_shadow_node_v0")

        self.model_path = Path(os.environ.get(
            "TRACER_PHASE_H2_OBJECTIVE_MODEL_JSON",
            "models/phase_h/h1_objective_selector_ridge_v0.json",
        ))
        self.suite = os.environ.get("TRACER_PHASE_H2_SUITE", "deployable_pretrain")

        self.context_topic = os.environ.get("TRACER_PHASE_H2_CONTEXT_TOPIC", "/tracer/terrain_context_label")
        self.odom_topic = os.environ.get("TRACER_PHASE_H2_ODOM_TOPIC", "/tracer/robot_odom_flat")
        self.ram_topic = os.environ.get("TRACER_PHASE_H2_RAM_TOPIC", "/tracer/ram_mismatch")
        self.ref_topic = os.environ.get("TRACER_PHASE_H2_REF_TOPIC", "/tracer/empirical_mpc_reference")
        self.beta_topic = os.environ.get("TRACER_PHASE_H2_OBJECTIVE_BETA_TOPIC", "/tracer/objective_beta_h2_shadow")

        self.window_s = env_float("TRACER_PHASE_H2_WINDOW_S", 8.0)
        self.pub_hz = env_float("TRACER_PHASE_H2_PUB_HZ", 10.0)
        self.default_context = os.environ.get("TRACER_PHASE_H2_DEFAULT_CONTEXT", "unknown")

        self.model = self._load_model()
        self.feature_cols = self.model["feature_cols"]
        self.feature_means = [float(x) for x in self.model["feature_means"]]
        self.feature_stds = [float(x) if abs(float(x)) > 1e-12 else 1.0 for x in self.model["feature_stds"]]
        self.W = self.model["coef_with_intercept"]

        if len(self.feature_cols) != len(self.feature_means) or len(self.feature_cols) != len(self.feature_stds):
            raise RuntimeError("H2 model feature metadata length mismatch")

        if len(self.W) != len(self.feature_cols) + 1:
            raise RuntimeError("H2 model coefficient shape mismatch")

        self.ctx = self.default_context
        self.x = None
        self.y = None
        self.ram = [None, None, None]
        self.ref = [None, None, None, None, None]  # vx, yaw, body_h, clearance, enable

        self.samples = deque()

        self.pub = self.create_publisher(Float64MultiArray, self.beta_topic, 10)

        self.create_subscription(String, self.context_topic, self.on_context, 10)
        self.create_subscription(Float64MultiArray, self.odom_topic, self.on_odom, 10)
        self.create_subscription(Float64MultiArray, self.ram_topic, self.on_ram, 10)
        self.create_subscription(Float64MultiArray, self.ref_topic, self.on_ref, 10)

        log_dir = Path(os.environ.get("TRACER_PHASE_H2_LOG_DIR", "logs/phase_h/h2_objective_shadow_latest"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = log_dir / "objective_selector_h2_shadow_v0.csv"
        self.log_f = open(self.log_csv, "w", newline="")
        self.writer = csv.DictWriter(
            self.log_f,
            fieldnames=[
                "t_wall",
                "suite",
                "context",
                "x",
                "y",
                "window_n",
                "observed_features",
                "imputed_features",
                "beta_motion",
                "beta_stability",
                "beta_energy",
                "raw_motion",
                "raw_stability",
                "raw_energy",
                "ram_slip_proxy",
                "ram_roughness_proxy",
                "ram_sigma",
                "ref_vx",
                "ref_yaw_rate",
                "ref_clearance",
            ],
        )
        self.writer.writeheader()
        self.log_f.flush()

        dt = 1.0 / max(self.pub_hz, 1e-6)
        self.timer = self.create_timer(dt, self.on_timer)

        self.get_logger().info(f"[TRACER:H2] loaded model={self.model_path} suite={self.suite}")
        self.get_logger().info(f"[TRACER:H2] publishing beta shadow to {self.beta_topic}")
        self.get_logger().info(f"[TRACER:H2] logging to {self.log_csv}")

    def _load_model(self):
        if not self.model_path.exists():
            raise RuntimeError(f"missing H2 model JSON: {self.model_path}")

        root = json.loads(self.model_path.read_text())
        suites = root.get("suites", {})
        if self.suite not in suites:
            raise RuntimeError(f"suite {self.suite} not found in {self.model_path}. available={list(suites)}")

        return suites[self.suite]["model"]

    def on_context(self, msg):
        self.ctx = context_key(msg.data)

    def on_odom(self, msg):
        data = list(msg.data)
        if len(data) >= 2:
            self.x = float(data[0])
            self.y = float(data[1])

    def on_ram(self, msg):
        data = list(msg.data)
        # expected proxy convention: [slip_proxy, roughness_proxy, sigma, ...]
        if len(data) >= 1:
            self.ram[0] = float(data[0])
        if len(data) >= 2:
            self.ram[1] = float(data[1])
        if len(data) >= 3:
            self.ram[2] = float(data[2])

    def on_ref(self, msg):
        data = list(msg.data)
        # expected ref: [seq, vx, yaw_rate, body_h, clearance, enable]
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

    def add_stats(self, feat, prefix, vals):
        m = fmean(vals)
        s = fstd(vals)
        mn = fmin(vals)
        mx = fmax(vals)

        if m is not None:
            feat[f"{prefix}_mean_g1"] = m
        if s is not None:
            feat[f"{prefix}_std_g1"] = s
        if mn is not None:
            feat[f"{prefix}_min_g1"] = mn
        if mx is not None:
            feat[f"{prefix}_max_g1"] = mx

    def snapshot(self, now):
        self.samples.append({
            "t": now,
            "ctx": context_key(self.ctx),
            "x": self.x,
            "y": self.y,
            "ram": list(self.ram),
            "ref": list(self.ref),
        })

        while self.samples and now - self.samples[0]["t"] > self.window_s:
            self.samples.popleft()

    def build_features(self):
        feat = {}

        samples = list(self.samples)
        if not samples:
            return feat

        # Context fractions.
        ctxs = [context_key(s["ctx"]) for s in samples]
        cnt = Counter(ctxs)
        n = float(len(ctxs))
        for k in ["flat", "start_flat", "rough", "upslope", "downslope", "goal_flat", "unknown"]:
            feat[f"context_frac_{k}"] = cnt.get(k, 0) / n
        feat["context_unique_count"] = float(len(cnt))

        # RAM window stats.
        slip = [s["ram"][0] for s in samples if finite(s["ram"][0])]
        rough = [s["ram"][1] for s in samples if finite(s["ram"][1])]
        sigma = [s["ram"][2] for s in samples if finite(s["ram"][2])]
        self.add_stats(feat, "ram_slip_proxy_mean", slip)
        self.add_stats(feat, "ram_roughness_proxy_mean", rough)
        self.add_stats(feat, "ram_sigma_mean", sigma)

        # Reference window stats.
        ref_vx = [s["ref"][0] for s in samples if finite(s["ref"][0])]
        ref_yaw = [s["ref"][1] for s in samples if finite(s["ref"][1])]
        ref_body = [s["ref"][2] for s in samples if finite(s["ref"][2])]
        ref_clear = [s["ref"][3] for s in samples if finite(s["ref"][3])]
        self.add_stats(feat, "ref_vx_mean", ref_vx)
        self.add_stats(feat, "ref_yaw_rate_mean", ref_yaw)
        self.add_stats(feat, "ref_body_h_mean", ref_body)
        self.add_stats(feat, "ref_clearance_mean", ref_clear)

        # Odom/lateral window features.
        odom = [s for s in samples if finite(s["x"]) and finite(s["y"])]
        if len(odom) >= 1:
            xs = [float(s["x"]) for s in odom]
            ys = [float(s["y"]) for s in odom]
            feat["x_progress_g1"] = xs[-1] - xs[0]
            feat["y_change_g1"] = ys[-1] - ys[0]
            feat["abs_y_change_g1"] = abs(ys[-1] - ys[0])

            ye, ym, yl = thirds(ys)
            aye, aym, ayl = thirds([abs(y) for y in ys])

            if ye:
                feat["y_early_mean_g1"] = fmean(ye)
                feat["abs_y_early_mean_g1"] = fmean(aye)
            if ym:
                feat["y_mid_mean_g1"] = fmean(ym)
                feat["abs_y_mid_mean_g1"] = fmean(aym)
            if yl:
                feat["y_late_mean_g1"] = fmean(yl)
                feat["abs_y_late_mean_g1"] = fmean(ayl)

            if ye and yl:
                feat["y_late_minus_early_g1"] = fmean(yl) - fmean(ye)
                feat["abs_y_late_minus_early_g1"] = fmean(ayl) - fmean(aye)

            dx = xs[-1] - xs[0]
            dy = ys[-1] - ys[0]
            feat["lateral_per_forward_g1"] = abs(dy) / max(abs(dx), 0.05)

            if len(odom) >= 2:
                dt = float(odom[-1]["t"] - odom[0]["t"])
                if dt > 1e-6:
                    vx_est = dx / dt
                    rvx = fmean(ref_vx)
                    if rvx is not None:
                        feat["vx_tracking_error_g1"] = vx_est - rvx
                        feat["vx_tracking_abs_error_g1"] = abs(vx_est - rvx)
                        feat["vx_tracking_ratio_g1"] = vx_est / rvx if abs(rvx) > 1e-4 else 0.0

        # Optional diagnostic reset_y override.
        reset_y_raw = os.environ.get("TRACER_PHASE_H2_RESET_Y", "")
        if reset_y_raw != "":
            try:
                reset_y = float(reset_y_raw)
                feat["reset_y_mean_g1"] = reset_y
                feat["reset_y_min_g1"] = reset_y
                feat["reset_y_max_g1"] = reset_y
            except Exception:
                pass

        return feat

    def predict_beta(self, feat):
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

        beta = project_simplex(raw)
        return raw, beta, observed, imputed

    def on_timer(self):
        now = time.time()
        self.snapshot(now)
        feat = self.build_features()
        raw, beta, observed, imputed = self.predict_beta(feat)

        msg = Float64MultiArray()
        msg.data = [float(beta[0]), float(beta[1]), float(beta[2])]
        self.pub.publish(msg)

        self.writer.writerow({
            "t_wall": f"{now:.6f}",
            "suite": self.suite,
            "context": context_key(self.ctx),
            "x": "" if self.x is None else f"{self.x:.6f}",
            "y": "" if self.y is None else f"{self.y:.6f}",
            "window_n": len(self.samples),
            "observed_features": observed,
            "imputed_features": imputed,
            "beta_motion": f"{beta[0]:.8f}",
            "beta_stability": f"{beta[1]:.8f}",
            "beta_energy": f"{beta[2]:.8f}",
            "raw_motion": f"{raw[0]:.8f}",
            "raw_stability": f"{raw[1]:.8f}",
            "raw_energy": f"{raw[2]:.8f}",
            "ram_slip_proxy": "" if self.ram[0] is None else f"{self.ram[0]:.6f}",
            "ram_roughness_proxy": "" if self.ram[1] is None else f"{self.ram[1]:.6f}",
            "ram_sigma": "" if self.ram[2] is None else f"{self.ram[2]:.6f}",
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
    node = H2ObjectiveSelectorShadow()
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
