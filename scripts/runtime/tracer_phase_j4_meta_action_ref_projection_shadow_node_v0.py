#!/usr/bin/env python3
import csv
import math
import os
import time
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


def clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))


def context_key(label):
    label = (label or "unknown").strip()
    known = {"flat", "start_flat", "rough", "upslope", "downslope", "goal_flat", "unknown"}
    return label if label in known else "unknown"


class J4MetaActionRefProjectionShadow(Node):
    def __init__(self):
        super().__init__("tracer_phase_j4_meta_action_ref_projection_shadow_node_v0")

        self.empirical_ref_topic = os.environ.get(
            "TRACER_PHASE_J4_EMPIRICAL_REF_TOPIC",
            "/tracer/empirical_mpc_reference",
        )
        self.theta_topic = os.environ.get(
            "TRACER_PHASE_J4_THETA_TOPIC",
            "/tracer/meta_action_theta_shadow",
        )
        self.context_topic = os.environ.get(
            "TRACER_PHASE_J4_CONTEXT_TOPIC",
            "/tracer/terrain_context_label",
        )
        self.odom_topic = os.environ.get(
            "TRACER_PHASE_J4_ODOM_TOPIC",
            "/tracer/robot_odom_flat",
        )
        self.out_topic = os.environ.get(
            "TRACER_PHASE_J4_PROJECTED_REF_TOPIC",
            "/tracer/meta_action_projected_ref_shadow",
        )

        self.pub_hz = env_float("TRACER_PHASE_J4_PUB_HZ", 10.0)

        # Optional message-driven projection.
        #
        # In timer mode, J4 and J7 may become phase-locked such that
        # J7 repeatedly compares empirical sequence N against projected
        # sequence N-1. Event-driven mode projects each newly received
        # empirical reference immediately.
        #
        # Disabled by default to preserve historical behavior.
        self.event_driven = (
            os.environ.get(
                "TRACER_PHASE_J4_EVENT_DRIVEN",
                "0",
            )
            == "1"
        )

        self.vx_min = env_float("TRACER_PHASE_J4_VX_MIN", 0.025)
        self.vx_max = env_float("TRACER_PHASE_J4_VX_MAX", 0.220)
        self.yaw_min = env_float("TRACER_PHASE_J4_YAW_MIN", -0.200)
        self.yaw_max = env_float("TRACER_PHASE_J4_YAW_MAX", 0.200)
        self.body_h_min = env_float("TRACER_PHASE_J4_BODY_H_MIN", 0.300)
        self.body_h_max = env_float("TRACER_PHASE_J4_BODY_H_MAX", 0.340)
        self.clearance_min = env_float("TRACER_PHASE_J4_CLEARANCE_MIN", 0.035)
        self.clearance_max = env_float("TRACER_PHASE_J4_CLEARANCE_MAX", 0.065)
        self.hold_vx = env_float("TRACER_PHASE_J4_HOLD_VX", 0.025)
        self.goal_x_threshold = env_float("TRACER_PHASE_J4_GOAL_X_THRESHOLD", 7.90)

        self.ctx = "unknown"
        self.x = None
        self.y = None

        self.emp_ref = None
        self.emp_ref_t = None

        self.theta = None
        self.theta_t = None

        self.pub = self.create_publisher(Float64MultiArray, self.out_topic, 10)

        self.create_subscription(Float64MultiArray, self.empirical_ref_topic, self.on_emp_ref, 10)
        self.create_subscription(Float64MultiArray, self.theta_topic, self.on_theta, 10)
        self.create_subscription(String, self.context_topic, self.on_context, 10)
        self.create_subscription(Float64MultiArray, self.odom_topic, self.on_odom, 10)

        log_dir = Path(os.environ.get("TRACER_PHASE_J4_LOG_DIR", "logs/phase_j/j4_ref_projection_shadow_latest"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = log_dir / "meta_action_ref_projection_shadow_j4_v0.csv"
        self.log_f = open(self.log_csv, "w", newline="")
        self.writer = csv.DictWriter(
            self.log_f,
            fieldnames=[
                "t_wall",
                "context",
                "x",
                "y",
                "action_id",
                "vx_scale",
                "vx_delta",
                "yaw_gain_scale",
                "body_h_delta",
                "clearance_delta",
                "stability_bias",
                "energy_bias",
                "hold_override",
                "hold_guard",
                "emp_seq",
                "emp_vx",
                "emp_yaw_rate",
                "emp_body_h",
                "emp_clearance",
                "emp_enable",
                "proj_seq",
                "proj_vx",
                "proj_yaw_rate",
                "proj_body_h",
                "proj_clearance",
                "proj_enable",
                "delta_vx",
                "delta_yaw_rate",
                "delta_body_h",
                "delta_clearance",
            ],
        )
        self.writer.writeheader()
        self.log_f.flush()

        self.timer = None

        if not self.event_driven:
            self.timer = self.create_timer(
                1.0 / max(self.pub_hz, 1e-6),
                self.on_timer,
            )

        self.get_logger().info(f"[TRACER:J4] empirical ref input={self.empirical_ref_topic}")
        self.get_logger().info(f"[TRACER:J4] theta input={self.theta_topic}")
        self.get_logger().info(f"[TRACER:J4] projected ref output={self.out_topic}")
        self.get_logger().info(f"[TRACER:J4] event_driven={self.event_driven}")
        self.get_logger().info(f"[TRACER:J4] logging to {self.log_csv}")
        self.get_logger().info("[TRACER:J4] shadow only: does not publish /tracer/mpc_reference")

    def on_context(self, msg):
        self.ctx = context_key(msg.data)

    def on_odom(self, msg):
        data = list(msg.data)
        if len(data) >= 2:
            self.x = float(data[0])
            self.y = float(data[1])

    def on_emp_ref(self, msg):
        data = list(msg.data)

        if len(data) >= 6:
            now = time.time()

            self.emp_ref = [
                float(v)
                for v in data[:6]
            ]
            self.emp_ref_t = now

            if (
                self.event_driven
                and self.theta is not None
            ):
                self.publish_projection(now)

    def on_theta(self, msg):
        data = list(msg.data)
        if len(data) >= 9:
            self.theta = [float(v) for v in data[:9]]
            self.theta_t = time.time()

    def project(self):
        seq, vx, yaw, body_h, clearance, enable = self.emp_ref

        (
            action_id,
            vx_scale,
            vx_delta,
            yaw_gain_scale,
            body_h_delta,
            clearance_delta,
            stability_bias,
            energy_bias,
            hold_override,
        ) = self.theta

        hold_guard = (
            hold_override >= 0.5
            or context_key(self.ctx) == "goal_flat"
            or (self.x is not None and self.x >= self.goal_x_threshold)
        )

        if hold_guard:
            proj_vx = self.hold_vx
            proj_yaw = 0.0
            proj_body_h = clamp(body_h + body_h_delta, self.body_h_min, self.body_h_max)
            proj_clearance = clamp(clearance + clearance_delta, self.clearance_min, self.clearance_max)
            proj_enable = 1.0
        else:
            proj_vx = clamp(vx * vx_scale + vx_delta, self.vx_min, self.vx_max)
            proj_yaw = clamp(yaw * yaw_gain_scale, self.yaw_min, self.yaw_max)
            proj_body_h = clamp(body_h + body_h_delta, self.body_h_min, self.body_h_max)
            proj_clearance = clamp(clearance + clearance_delta, self.clearance_min, self.clearance_max)
            proj_enable = 1.0 if enable >= 0.5 else 0.0

        return {
            "action_id": action_id,
            "vx_scale": vx_scale,
            "vx_delta": vx_delta,
            "yaw_gain_scale": yaw_gain_scale,
            "body_h_delta": body_h_delta,
            "clearance_delta": clearance_delta,
            "stability_bias": stability_bias,
            "energy_bias": energy_bias,
            "hold_override": hold_override,
            "hold_guard": hold_guard,
            "proj": [seq, proj_vx, proj_yaw, proj_body_h, proj_clearance, proj_enable],
        }

    def publish_projection(self, now=None):
        if self.emp_ref is None or self.theta is None:
            return

        if now is None:
            now = time.time()
        out = self.project()
        proj = out["proj"]

        msg = Float64MultiArray()
        msg.data = [float(v) for v in proj]
        self.pub.publish(msg)

        emp = self.emp_ref

        self.writer.writerow({
            "t_wall": f"{now:.6f}",
            "context": context_key(self.ctx),
            "x": "" if self.x is None else f"{self.x:.6f}",
            "y": "" if self.y is None else f"{self.y:.6f}",
            "action_id": f"{out['action_id']:.0f}",
            "vx_scale": f"{out['vx_scale']:.8f}",
            "vx_delta": f"{out['vx_delta']:.8f}",
            "yaw_gain_scale": f"{out['yaw_gain_scale']:.8f}",
            "body_h_delta": f"{out['body_h_delta']:.8f}",
            "clearance_delta": f"{out['clearance_delta']:.8f}",
            "stability_bias": f"{out['stability_bias']:.8f}",
            "energy_bias": f"{out['energy_bias']:.8f}",
            "hold_override": "1" if out["hold_override"] >= 0.5 else "0",
            "hold_guard": "1" if out["hold_guard"] else "0",
            "emp_seq": f"{emp[0]:.0f}",
            "emp_vx": f"{emp[1]:.8f}",
            "emp_yaw_rate": f"{emp[2]:.8f}",
            "emp_body_h": f"{emp[3]:.8f}",
            "emp_clearance": f"{emp[4]:.8f}",
            "emp_enable": f"{emp[5]:.8f}",
            "proj_seq": f"{proj[0]:.0f}",
            "proj_vx": f"{proj[1]:.8f}",
            "proj_yaw_rate": f"{proj[2]:.8f}",
            "proj_body_h": f"{proj[3]:.8f}",
            "proj_clearance": f"{proj[4]:.8f}",
            "proj_enable": f"{proj[5]:.8f}",
            "delta_vx": f"{proj[1] - emp[1]:.8f}",
            "delta_yaw_rate": f"{proj[2] - emp[2]:.8f}",
            "delta_body_h": f"{proj[3] - emp[3]:.8f}",
            "delta_clearance": f"{proj[4] - emp[4]:.8f}",
        })
        self.log_f.flush()

    def on_timer(self):
        self.publish_projection()


    def destroy_node(self):
        try:
            self.log_f.flush()
            self.log_f.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = J4MetaActionRefProjectionShadow()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if e.__class__.__name__ != "ExternalShutdownException":
            raise
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
