#!/usr/bin/env python3
import csv
import json
import math
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int32, String


def env_float(name, default):
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def ff(x, default=None):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def context_key(label):
    label = (label or "unknown").strip()
    known = {"flat", "start_flat", "rough", "upslope", "downslope", "goal_flat", "unknown"}
    return label if label in known else "unknown"


class J3MetaActionShadow(Node):
    def __init__(self):
        super().__init__("tracer_phase_j3_meta_action_shadow_node_v0")

        self.policy_json = Path(os.environ.get(
            "TRACER_PHASE_J3_POLICY_JSON",
            "models/phase_j/j2_meta_action_tabular_policy_v0.json",
        ))
        self.bank_json = Path(os.environ.get(
            "TRACER_PHASE_J3_BANK_JSON",
            "configs/phase_j/meta_action_bank_v0.json",
        ))

        self.context_topic = os.environ.get("TRACER_PHASE_J3_CONTEXT_TOPIC", "/tracer/terrain_context_label")
        self.odom_topic = os.environ.get("TRACER_PHASE_J3_ODOM_TOPIC", "/tracer/robot_odom_flat")
        self.beta_topic = os.environ.get("TRACER_PHASE_J3_BETA_TOPIC", "/tracer/objective_beta")
        self.ram_topic = os.environ.get("TRACER_PHASE_J3_RAM_TOPIC", "/tracer/ram_mismatch")

        self.action_id_topic = os.environ.get("TRACER_PHASE_J3_ACTION_ID_TOPIC", "/tracer/meta_action_id_shadow")
        self.action_name_topic = os.environ.get("TRACER_PHASE_J3_ACTION_NAME_TOPIC", "/tracer/meta_action_name_shadow")
        self.theta_topic = os.environ.get("TRACER_PHASE_J3_THETA_TOPIC", "/tracer/meta_action_theta_shadow")

        self.pub_hz = env_float("TRACER_PHASE_J3_PUB_HZ", 10.0)

        self.policy = json.loads(self.policy_json.read_text())
        self.bank = json.loads(self.bank_json.read_text())

        self.actions = {a["name"]: a for a in self.bank["actions"]}
        self.actions_by_id = {int(a["id"]): a for a in self.bank["actions"]}

        self.ctx = "unknown"
        self.x = None
        self.y = None
        self.beta = None
        self.beta_t = None
        self.ram = None
        self.ram_t = None

        self.pub_action_id = self.create_publisher(Int32, self.action_id_topic, 10)
        self.pub_action_name = self.create_publisher(String, self.action_name_topic, 10)
        self.pub_theta = self.create_publisher(Float64MultiArray, self.theta_topic, 10)

        self.create_subscription(String, self.context_topic, self.on_context, 10)
        self.create_subscription(Float64MultiArray, self.odom_topic, self.on_odom, 10)
        self.create_subscription(Float64MultiArray, self.beta_topic, self.on_beta, 10)
        self.create_subscription(Float64MultiArray, self.ram_topic, self.on_ram, 10)

        log_dir = Path(os.environ.get("TRACER_PHASE_J3_LOG_DIR", "logs/phase_j/j3_meta_action_shadow_latest"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = log_dir / "meta_action_shadow_j3_v0.csv"
        self.log_f = open(self.log_csv, "w", newline="")
        self.writer = csv.DictWriter(
            self.log_f,
            fieldnames=[
                "t_wall",
                "context",
                "x",
                "y",
                "beta_motion",
                "beta_stability",
                "beta_energy",
                "ram_slip",
                "ram_rough",
                "ram_sigma",
                "action_id",
                "action_name",
                "reason",
                "vx_scale",
                "vx_delta",
                "yaw_gain_scale",
                "body_h_delta",
                "clearance_delta",
                "stability_bias",
                "energy_bias",
                "hold_override",
            ],
        )
        self.writer.writeheader()
        self.log_f.flush()

        self.timer = self.create_timer(1.0 / max(self.pub_hz, 1e-6), self.on_timer)

        self.get_logger().info(f"[TRACER:J3] policy={self.policy_json}")
        self.get_logger().info(f"[TRACER:J3] bank={self.bank_json}")
        self.get_logger().info(f"[TRACER:J3] beta input={self.beta_topic}")
        self.get_logger().info(f"[TRACER:J3] RAM input={self.ram_topic}")
        self.get_logger().info(f"[TRACER:J3] publishing action id to {self.action_id_topic}")
        self.get_logger().info(f"[TRACER:J3] publishing action name to {self.action_name_topic}")
        self.get_logger().info(f"[TRACER:J3] publishing theta to {self.theta_topic}")
        self.get_logger().info(f"[TRACER:J3] logging to {self.log_csv}")
        self.get_logger().info("[TRACER:J3] shadow only: does not publish /tracer/mpc_reference")

    def on_context(self, msg):
        self.ctx = context_key(msg.data)

    def on_odom(self, msg):
        data = list(msg.data)
        if len(data) >= 2:
            self.x = float(data[0])
            self.y = float(data[1])

    def on_beta(self, msg):
        data = list(msg.data)
        if len(data) >= 3:
            s = sum(float(v) for v in data[:3])
            if abs(s) > 1e-12:
                self.beta = [float(data[0]) / s, float(data[1]) / s, float(data[2]) / s]
            else:
                self.beta = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            self.beta_t = time.time()

    def on_ram(self, msg):
        data = list(msg.data)
        if len(data) >= 3:
            self.ram = [float(data[0]), float(data[1]), float(data[2])]
            self.ram_t = time.time()

    def select_action(self):
        guards = self.policy["guards"]

        ctx = context_key(self.ctx)
        x = self.x
        y = self.y
        abs_y = abs(y) if y is not None else 0.0

        if ctx == "goal_flat":
            return "goal_hold", "goal_context_guard"

        if x is not None and x >= float(guards["goal_x_threshold"]):
            return "goal_hold", "goal_x_guard"

        if (
            abs_y >= float(guards["lateral_abs_y_threshold"])
            and ctx in guards["lateral_guard_contexts"]
        ):
            return "lateral_recovery_soft", "lateral_guard"

        cp = self.policy.get("context_policy", {}).get(ctx)
        if cp:
            return cp["top_action_name"], "context_top1"

        return self.policy["fallback_action_name"], "fallback"

    def on_timer(self):
        now = time.time()
        action_name, reason = self.select_action()
        action = self.actions.get(action_name)
        if action is None:
            action_name = self.policy["fallback_action_name"]
            action = self.actions[action_name]
            reason = "fallback_missing_action"

        action_id = int(action["id"])
        th = action["theta"]

        msg_id = Int32()
        msg_id.data = action_id
        self.pub_action_id.publish(msg_id)

        msg_name = String()
        msg_name.data = action_name
        self.pub_action_name.publish(msg_name)

        msg_theta = Float64MultiArray()
        msg_theta.data = [
            float(action_id),
            float(th.get("vx_scale", 1.0)),
            float(th.get("vx_delta", 0.0)),
            float(th.get("yaw_gain_scale", 1.0)),
            float(th.get("body_h_delta", 0.0)),
            float(th.get("clearance_delta", 0.0)),
            float(th.get("stability_bias", 0.0)),
            float(th.get("energy_bias", 0.0)),
            1.0 if bool(th.get("hold_override", False)) else 0.0,
        ]
        self.pub_theta.publish(msg_theta)

        beta = self.beta or [None, None, None]
        ram = self.ram or [None, None, None]

        self.writer.writerow({
            "t_wall": f"{now:.6f}",
            "context": context_key(self.ctx),
            "x": "" if self.x is None else f"{self.x:.6f}",
            "y": "" if self.y is None else f"{self.y:.6f}",
            "beta_motion": "" if beta[0] is None else f"{beta[0]:.8f}",
            "beta_stability": "" if beta[1] is None else f"{beta[1]:.8f}",
            "beta_energy": "" if beta[2] is None else f"{beta[2]:.8f}",
            "ram_slip": "" if ram[0] is None else f"{ram[0]:.8f}",
            "ram_rough": "" if ram[1] is None else f"{ram[1]:.8f}",
            "ram_sigma": "" if ram[2] is None else f"{ram[2]:.8f}",
            "action_id": action_id,
            "action_name": action_name,
            "reason": reason,
            "vx_scale": f"{float(th.get('vx_scale', 1.0)):.8f}",
            "vx_delta": f"{float(th.get('vx_delta', 0.0)):.8f}",
            "yaw_gain_scale": f"{float(th.get('yaw_gain_scale', 1.0)):.8f}",
            "body_h_delta": f"{float(th.get('body_h_delta', 0.0)):.8f}",
            "clearance_delta": f"{float(th.get('clearance_delta', 0.0)):.8f}",
            "stability_bias": f"{float(th.get('stability_bias', 0.0)):.8f}",
            "energy_bias": f"{float(th.get('energy_bias', 0.0)):.8f}",
            "hold_override": "1" if bool(th.get("hold_override", False)) else "0",
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
    node = J3MetaActionShadow()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # timeout can trigger external shutdown in some local shells; keep real errors visible.
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
