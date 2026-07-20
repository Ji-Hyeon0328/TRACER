#!/usr/bin/env python3
import csv
import json
import math
import os
import signal
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
try:
    from rclpy.exceptions import RCLError
except Exception:
    class RCLError(Exception):
        pass

from std_msgs.msg import Float64MultiArray, String


def clamp(x, lo=0.0, hi=1.0):
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(lo, min(hi, x))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def max_abs(xs):
    xs = list(xs)
    return max((abs(float(x)) for x in xs), default=0.0)


def normalize_beta(v):
    vals = [max(1e-9, float(x)) for x in v]
    s = sum(vals)
    return [x / s for x in vals]


CONTEXT_PRIOR_BETA = {
    "flat":       (0.45, 0.25, 0.30),
    "start_flat": (0.45, 0.25, 0.30),
    "rough":      (0.25, 0.55, 0.20),
    "upslope":    (0.35, 0.40, 0.25),
    "downslope":  (0.25, 0.60, 0.15),
    "goal_flat":  (0.20, 0.50, 0.30),
    "unknown":    (0.33, 0.34, 0.33),
}


class D7ObjectiveSelectorShadow(Node):
    def __init__(self):
        super().__init__("tracer_phase_d7_objective_selector_shadow_v0")

        self.model_json = os.environ.get(
            "TRACER_PHASE_D7_OBJECTIVE_MODEL_JSON",
            "models/phase_d7/d7_objective_selector_ridge_v0.json",
        )
        self.out_topic = os.environ.get(
            "TRACER_PHASE_D7_OBJECTIVE_BETA_TOPIC",
            "/tracer/objective_beta_shadow",
        )
        self.actual_beta_topic = os.environ.get(
            "TRACER_PHASE_D7_ACTUAL_BETA_TOPIC",
            "/tracer/objective_beta",
        )
        self.ref_topic = os.environ.get(
            "TRACER_PHASE_D7_REF_TOPIC",
            os.environ.get("TRACER_REF_TOPIC", "/tracer/empirical_mpc_reference"),
        )
        self.odom_topic = os.environ.get(
            "TRACER_PHASE_D7_ODOM_TOPIC",
            "/tracer/robot_odom_flat",
        )
        self.context_topic = os.environ.get(
            "TRACER_PHASE_D7_CONTEXT_TOPIC",
            "/tracer/terrain_context_label",
        )
        self.ram_topic = os.environ.get(
            "TRACER_PHASE_D7_RAM_TOPIC",
            "/tracer/ram_mismatch",
        )

        self.goal_x = float(os.environ.get("TRACER_PHASE_D7_GOAL_X", "8.0"))
        self.lateral_bound = float(os.environ.get("TRACER_PHASE_D7_LATERAL_BOUND", "2.0"))
        self.stop_margin = float(os.environ.get("TRACER_PHASE_D7_STOP_MARGIN", "-0.05"))
        self.window_n = int(os.environ.get("TRACER_PHASE_D7_WINDOW_N", "300"))
        self.pub_hz = float(os.environ.get("TRACER_PHASE_D7_PUB_HZ", "10.0"))

        # Runtime-safe post-processing for shadow beta.
        # The ridge model can produce out-of-distribution raw beta during online rollout.
        # Keep raw beta in logs, but publish a floor/prior-blended normalized beta.
        self.beta_floor = float(os.environ.get("TRACER_PHASE_D7_BETA_FLOOR", "0.05"))
        self.prior_blend = float(os.environ.get("TRACER_PHASE_D7_PRIOR_BLEND", "0.20"))

        self.log_csv = os.environ.get("TRACER_PHASE_D7_SHADOW_LOG_CSV", "")
        if not self.log_csv:
            log_dir = os.environ.get("TRACER_PHASE_D5_LOG_DIR", "logs/phase_d7_shadow")
            self.log_csv = str(Path(log_dir) / "d7_objective_selector_shadow_v0.csv")
        Path(self.log_csv).parent.mkdir(parents=True, exist_ok=True)

        self.model = json.loads(Path(self.model_json).read_text())
        self.contexts = self.model["contexts"]
        self.features = self.model["features"]
        self.weights = self.model["weights"]

        self.current_context = "unknown"
        self.x = 0.0
        self.y = 0.0
        self.reset_y = None

        self.ram_slip = 0.0
        self.ram_rough = 0.0
        self.ram_sigma = 0.0

        self.ref_seq = 0.0
        self.ref_vx = 0.0
        self.ref_yaw = 0.0
        self.ref_body_h = 0.32
        self.ref_clearance = 0.045
        self.ref_enable = 1.0

        self.actual_beta = [float("nan"), float("nan"), float("nan")]

        self.y_window = deque(maxlen=self.window_n)
        self.abs_y_window = deque(maxlen=self.window_n)
        self.ref_vx_window = deque(maxlen=self.window_n)
        self.ref_yaw_window = deque(maxlen=self.window_n)
        self.ref_clearance_window = deque(maxlen=self.window_n)
        self.context_y = defaultdict(lambda: deque(maxlen=self.window_n))

        self.goal_reached = False
        self.x_at_goal = None
        self.max_x_after_goal = None

        self.pub = self.create_publisher(Float64MultiArray, self.out_topic, 10)

        self.create_subscription(String, self.context_topic, self.on_context, 10)
        self.create_subscription(Float64MultiArray, self.odom_topic, self.on_odom, 20)
        self.create_subscription(Float64MultiArray, self.ram_topic, self.on_ram, 10)
        self.create_subscription(Float64MultiArray, self.ref_topic, self.on_ref, 20)
        self.create_subscription(Float64MultiArray, self.actual_beta_topic, self.on_actual_beta, 10)

        self.csv_fp = open(self.log_csv, "w", newline="")
        self.csv_writer = csv.DictWriter(
            self.csv_fp,
            fieldnames=[
                "t_wall",
                "context",
                "x",
                "y",
                "reset_y",
                "pred_beta_motion",
                "pred_beta_stability",
                "pred_beta_energy",
                "raw_beta_motion",
                "raw_beta_stability",
                "raw_beta_energy",
                "prior_beta_motion",
                "prior_beta_stability",
                "prior_beta_energy",
                "actual_beta_motion",
                "actual_beta_stability",
                "actual_beta_energy",
                "err_beta_motion",
                "err_beta_stability",
                "err_beta_energy",
                "motion_score",
                "stability_score",
                "energy_score",
                "effort_proxy",
                "hold_drift",
                "rollout_max_abs_y",
                "rollout_mean_abs_y",
                "mean_abs_y_context",
                "max_abs_y_context",
                "ram_slip_proxy_mean",
                "ram_roughness_proxy_mean",
                "ram_sigma_mean",
                "ref_vx_mean",
                "ref_yaw_rate_mean",
                "ref_clearance_mean",
                "out_topic",
                "model_json",
            ],
        )
        self.csv_writer.writeheader()

        self.timer = self.create_timer(1.0 / max(self.pub_hz, 1e-6), self.on_timer)

        self.get_logger().info(f"D7 objective selector shadow model={self.model_json}")
        self.get_logger().info(f"publish shadow beta -> {self.out_topic}")
        self.get_logger().info(f"actual beta topic   -> {self.actual_beta_topic}")
        self.get_logger().info(f"ref topic           -> {self.ref_topic}")
        self.get_logger().info(f"log csv             -> {self.log_csv}")
        self.get_logger().info(f"beta floor/blend    -> floor={self.beta_floor}, prior_blend={self.prior_blend}")

    def on_context(self, msg):
        c = (msg.data or "unknown").strip()
        self.current_context = c if c else "unknown"

    def on_odom(self, msg):
        data = list(msg.data)
        if len(data) >= 2:
            self.x = float(data[0])
            self.y = float(data[1])
            if self.reset_y is None:
                self.reset_y = self.y

            self.y_window.append(self.y)
            self.abs_y_window.append(abs(self.y))
            self.context_y[self.current_context].append(self.y)

            goal_threshold = self.goal_x - self.stop_margin
            if self.x >= goal_threshold and not self.goal_reached:
                self.goal_reached = True
                self.x_at_goal = self.x
                self.max_x_after_goal = self.x
            if self.goal_reached:
                self.max_x_after_goal = max(self.max_x_after_goal or self.x, self.x)

    def on_ram(self, msg):
        data = list(msg.data)
        if len(data) >= 1:
            self.ram_slip = float(data[0])
        if len(data) >= 2:
            self.ram_rough = float(data[1])
        if len(data) >= 3:
            self.ram_sigma = float(data[2])

    def on_ref(self, msg):
        data = list(msg.data)
        if len(data) >= 6:
            self.ref_seq = float(data[0])
            self.ref_vx = float(data[1])
            self.ref_yaw = float(data[2])
            self.ref_body_h = float(data[3])
            self.ref_clearance = float(data[4])
            self.ref_enable = float(data[5])
        elif len(data) >= 5:
            self.ref_vx = float(data[0])
            self.ref_yaw = float(data[1])
            self.ref_body_h = float(data[2])
            self.ref_clearance = float(data[3])
            self.ref_enable = float(data[4])

        self.ref_vx_window.append(self.ref_vx)
        self.ref_yaw_window.append(self.ref_yaw)
        self.ref_clearance_window.append(self.ref_clearance)

    def on_actual_beta(self, msg):
        data = list(msg.data)
        if len(data) >= 3:
            self.actual_beta = [float(data[0]), float(data[1]), float(data[2])]

    def predict(self, feature_vec):
        xb = [1.0] + list(feature_vec)
        y = []
        for j in range(len(self.weights[0])):
            yj = 0.0
            for i in range(min(len(xb), len(self.weights))):
                yj += xb[i] * float(self.weights[i][j])
            y.append(yj)
        return normalize_beta(y)

    def postprocess_beta(self, raw_beta):
        floored = normalize_beta([max(self.beta_floor, b) for b in raw_beta])
        prior = list(CONTEXT_PRIOR_BETA.get(self.current_context, CONTEXT_PRIOR_BETA["unknown"]))
        beta = normalize_beta([
            (1.0 - self.prior_blend) * floored[i] + self.prior_blend * prior[i]
            for i in range(3)
        ])
        return beta, floored, prior

    def compute_features(self):
        ctx_y_vals = list(self.context_y[self.current_context])
        y_vals = list(self.y_window)

        ref_vx_mean = mean(self.ref_vx_window) if self.ref_vx_window else self.ref_vx
        ref_yaw_mean = mean(self.ref_yaw_window) if self.ref_yaw_window else self.ref_yaw
        ref_clearance_mean = mean(self.ref_clearance_window) if self.ref_clearance_window else self.ref_clearance

        rollout_max_abs_y = max_abs(y_vals) if y_vals else abs(self.y)
        rollout_mean_abs_y = mean(abs(v) for v in y_vals) if y_vals else abs(self.y)

        mean_abs_y_context = mean(abs(v) for v in ctx_y_vals) if ctx_y_vals else abs(self.y)
        max_abs_y_context = max_abs(ctx_y_vals) if ctx_y_vals else abs(self.y)

        if self.goal_reached and self.x_at_goal is not None:
            hold_drift = abs((self.max_x_after_goal or self.x) - self.x_at_goal)
        else:
            hold_drift = 0.0

        progress_score = clamp(self.x / max(self.goal_x, 1e-6))
        goal_score = 1.0 if self.goal_reached else 0.0
        speed_score = clamp(ref_vx_mean / 0.21)

        motion_score = clamp(0.60 * progress_score + 0.25 * goal_score + 0.15 * speed_score)

        group_lateral_score = 1.0 - clamp(max_abs_y_context / max(self.lateral_bound, 1e-6))
        rollout_lateral_score = 1.0 - clamp(rollout_max_abs_y / max(self.lateral_bound, 1e-6))
        hold_score = 1.0 - clamp(hold_drift / 0.25)
        stability_score = clamp(
            0.45 * group_lateral_score
            + 0.35 * rollout_lateral_score
            + 0.20 * hold_score
        )

        vx_effort = clamp(abs(ref_vx_mean) / 0.24)
        yaw_effort = clamp(abs(ref_yaw_mean) / 0.20)
        clearance_effort = clamp((ref_clearance_mean - 0.035) / max(1e-6, 0.065 - 0.035))
        effort_proxy = clamp(0.45 * vx_effort + 0.25 * yaw_effort + 0.30 * clearance_effort)
        energy_score = 1.0 - effort_proxy

        values = {
            "reset_y": self.reset_y if self.reset_y is not None else self.y,
            "y_mean": mean(y_vals) if y_vals else self.y,
            "mean_abs_y_context": mean_abs_y_context,
            "max_abs_y_context": max_abs_y_context,
            "ram_slip_proxy_mean": self.ram_slip,
            "ram_roughness_proxy_mean": self.ram_rough,
            "ram_sigma_mean": self.ram_sigma,
            "motion_score": motion_score,
            "stability_score": stability_score,
            "energy_score": energy_score,
            "effort_proxy": effort_proxy,
            "hold_drift": hold_drift,
            "rollout_max_abs_y": rollout_max_abs_y,
            "rollout_mean_abs_y": rollout_mean_abs_y,
        }

        feature_vec = [1.0 if self.current_context == c else 0.0 for c in self.contexts]
        feature_vec += [float(values.get(k, 0.0)) for k in self.features]
        return feature_vec, values

    def on_timer(self):
        feature_vec, values = self.compute_features()
        raw_beta = self.predict(feature_vec)
        beta, floored_beta, prior_beta = self.postprocess_beta(raw_beta)

        msg = Float64MultiArray()
        msg.data = beta
        self.pub.publish(msg)

        err = []
        for i in range(3):
            if math.isfinite(self.actual_beta[i]):
                err.append(beta[i] - self.actual_beta[i])
            else:
                err.append(float("nan"))

        self.csv_writer.writerow({
            "t_wall": f"{time.time():.6f}",
            "context": self.current_context,
            "x": f"{self.x:.9f}",
            "y": f"{self.y:.9f}",
            "reset_y": f"{(self.reset_y if self.reset_y is not None else self.y):.9f}",
            "pred_beta_motion": f"{beta[0]:.9f}",
            "pred_beta_stability": f"{beta[1]:.9f}",
            "pred_beta_energy": f"{beta[2]:.9f}",
            "raw_beta_motion": f"{raw_beta[0]:.9f}",
            "raw_beta_stability": f"{raw_beta[1]:.9f}",
            "raw_beta_energy": f"{raw_beta[2]:.9f}",
            "prior_beta_motion": f"{prior_beta[0]:.9f}",
            "prior_beta_stability": f"{prior_beta[1]:.9f}",
            "prior_beta_energy": f"{prior_beta[2]:.9f}",
            "actual_beta_motion": f"{self.actual_beta[0]:.9f}" if math.isfinite(self.actual_beta[0]) else "",
            "actual_beta_stability": f"{self.actual_beta[1]:.9f}" if math.isfinite(self.actual_beta[1]) else "",
            "actual_beta_energy": f"{self.actual_beta[2]:.9f}" if math.isfinite(self.actual_beta[2]) else "",
            "err_beta_motion": f"{err[0]:.9f}" if math.isfinite(err[0]) else "",
            "err_beta_stability": f"{err[1]:.9f}" if math.isfinite(err[1]) else "",
            "err_beta_energy": f"{err[2]:.9f}" if math.isfinite(err[2]) else "",
            "motion_score": f"{values['motion_score']:.9f}",
            "stability_score": f"{values['stability_score']:.9f}",
            "energy_score": f"{values['energy_score']:.9f}",
            "effort_proxy": f"{values['effort_proxy']:.9f}",
            "hold_drift": f"{values['hold_drift']:.9f}",
            "rollout_max_abs_y": f"{values['rollout_max_abs_y']:.9f}",
            "rollout_mean_abs_y": f"{values['rollout_mean_abs_y']:.9f}",
            "mean_abs_y_context": f"{values['mean_abs_y_context']:.9f}",
            "max_abs_y_context": f"{values['max_abs_y_context']:.9f}",
            "ram_slip_proxy_mean": f"{self.ram_slip:.9f}",
            "ram_roughness_proxy_mean": f"{self.ram_rough:.9f}",
            "ram_sigma_mean": f"{self.ram_sigma:.9f}",
            "ref_vx_mean": f"{mean(self.ref_vx_window) if self.ref_vx_window else self.ref_vx:.9f}",
            "ref_yaw_rate_mean": f"{mean(self.ref_yaw_window) if self.ref_yaw_window else self.ref_yaw:.9f}",
            "ref_clearance_mean": f"{mean(self.ref_clearance_window) if self.ref_clearance_window else self.ref_clearance:.9f}",
            "out_topic": self.out_topic,
            "model_json": self.model_json,
        })
        self.csv_fp.flush()

    def destroy_node(self):
        try:
            self.csv_fp.flush()
            self.csv_fp.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init(args=remove_ros_args())
    node = D7ObjectiveSelectorShadow()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
