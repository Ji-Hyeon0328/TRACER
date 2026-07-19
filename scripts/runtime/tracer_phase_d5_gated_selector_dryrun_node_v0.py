#!/usr/bin/env python3
import csv
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float64MultiArray, String


class GatedSelectorDryRun(Node):
    def __init__(self):
        super().__init__("tracer_phase_d5_gated_selector_dryrun_v0")

        ts = time.strftime("%Y%m%d_%H%M%S")
        default_log = f"logs/phase_d5_gate_dryrun_{ts}/gated_selector_dryrun_v0.csv"
        self.log_path = Path(os.environ.get("TRACER_PHASE_D5_GATE_LOG", default_log))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.context_topic = os.environ.get("TRACER_CONTEXT_LABEL_TOPIC", "/tracer/terrain_context_label")
        self.empirical_ref_topic = os.environ.get("TRACER_PHASE_D5_EMPIRICAL_REF_TOPIC", "/tracer/mpc_reference")
        self.learned_ref_topic = os.environ.get("TRACER_PHASE_D5_LEARNED_SHADOW_TOPIC", "/tracer/learned_selector_shadow_ref")
        self.gated_shadow_topic = os.environ.get("TRACER_PHASE_D5_GATED_SHADOW_TOPIC", "/tracer/gated_selector_dryrun_ref")

        self.vx_tol = float(os.environ.get("TRACER_PHASE_D5_GATE_VX_TOL", "0.020"))
        self.yaw_tol = float(os.environ.get("TRACER_PHASE_D5_GATE_YAW_TOL", "0.010"))
        self.body_h_tol = float(os.environ.get("TRACER_PHASE_D5_GATE_BODY_H_TOL", "0.010"))
        self.clearance_tol = float(os.environ.get("TRACER_PHASE_D5_GATE_CLEARANCE_TOL", "0.010"))
        self.enable_tol = float(os.environ.get("TRACER_PHASE_D5_GATE_ENABLE_TOL", "0.250"))
        self.hold_vx_threshold = float(os.environ.get("TRACER_PHASE_D5_GATE_HOLD_VX_THRESHOLD", "0.050"))

        # Dimension-selective gating.
        # For D6 MLP control, we usually trust learned vx/yaw/clearance,
        # but preserve empirical body_h/enable for safety.
        self.use_learned_vx = os.environ.get("TRACER_PHASE_D5_GATE_USE_LEARNED_VX", "1") == "1"
        self.use_learned_yaw = os.environ.get("TRACER_PHASE_D5_GATE_USE_LEARNED_YAW", "1") == "1"
        self.use_learned_body_h = os.environ.get("TRACER_PHASE_D5_GATE_USE_LEARNED_BODY_H", "1") == "1"
        self.use_learned_clearance = os.environ.get("TRACER_PHASE_D5_GATE_USE_LEARNED_CLEARANCE", "1") == "1"
        self.use_learned_enable = os.environ.get("TRACER_PHASE_D5_GATE_USE_LEARNED_ENABLE", "1") == "1"

        self.pub_hz = float(os.environ.get("TRACER_PHASE_D5_GATE_PUB_HZ", "10.0"))

        self.current_context = "unknown"
        self.seq = 0
        self.empirical = None
        self.learned = None

        self.create_subscription(String, self.context_topic, self.on_context, 10)
        self.create_subscription(Float64MultiArray, self.empirical_ref_topic, self.on_empirical, 10)
        self.create_subscription(Float64MultiArray, self.learned_ref_topic, self.on_learned, 10)

        self.pub = self.create_publisher(Float64MultiArray, self.gated_shadow_topic, 10)

        self.f = self.log_path.open("w", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=[
            "t_wall",
            "context",
            "gate_accept",
            "reject_reason",
            "emp_seq",
            "learned_seq",
            "selected_source",
            "emp_vx",
            "learned_vx",
            "selected_vx",
            "err_vx",
            "emp_yaw_rate",
            "learned_yaw_rate",
            "selected_yaw_rate",
            "err_yaw_rate",
            "emp_body_h",
            "learned_body_h",
            "selected_body_h",
            "err_body_h",
            "emp_clearance",
            "learned_clearance",
            "selected_clearance",
            "err_clearance",
            "emp_enable",
            "learned_enable",
            "selected_enable",
            "err_enable",
        ])
        self.writer.writeheader()

        self.timer = self.create_timer(1.0 / self.pub_hz, self.on_timer)

        self.get_logger().info(f"empirical_ref_topic={self.empirical_ref_topic}")
        self.get_logger().info(f"learned_ref_topic={self.learned_ref_topic}")
        self.get_logger().info(f"gated_shadow_topic={self.gated_shadow_topic}")
        self.get_logger().info(
            f"tolerances: vx={self.vx_tol}, yaw={self.yaw_tol}, "
            f"body_h={self.body_h_tol}, clearance={self.clearance_tol}, "
            f"enable={self.enable_tol}, hold_vx_threshold={self.hold_vx_threshold}"
        )
        self.get_logger().info(
            "learned_dims: "
            f"vx={self.use_learned_vx}, "
            f"yaw={self.use_learned_yaw}, "
            f"body_h={self.use_learned_body_h}, "
            f"clearance={self.use_learned_clearance}, "
            f"enable={self.use_learned_enable}"
        )
        self.get_logger().info(f"log_path={self.log_path}")

    def on_context(self, msg):
        self.current_context = msg.data.strip() or "unknown"

    @staticmethod
    def normalize_ref(data):
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

    def on_empirical(self, msg):
        self.empirical = self.normalize_ref(msg.data)

    def on_learned(self, msg):
        self.learned = self.normalize_ref(msg.data)

    def decide(self):
        if self.empirical is None:
            return False, "missing_empirical"
        if self.learned is None:
            return False, "missing_learned"

        e = self.empirical
        l = self.learned

        # Safety rule: during goal/hold phase, keep empirical hold behavior.
        # Learned yaw can remain nonzero because it is fitted from y, but the
        # real controller should preserve the stable hold reference.
        if self.current_context == "goal_flat" or abs(e["vx"]) <= self.hold_vx_threshold:
            return False, "hold_phase_empirical"

        checks = []
        if self.use_learned_vx:
            checks.append(("vx", abs(l["vx"] - e["vx"]), self.vx_tol))
        if self.use_learned_yaw:
            checks.append(("yaw_rate", abs(l["yaw_rate"] - e["yaw_rate"]), self.yaw_tol))
        if self.use_learned_body_h:
            checks.append(("body_h", abs(l["body_h"] - e["body_h"]), self.body_h_tol))
        if self.use_learned_clearance:
            checks.append(("clearance", abs(l["clearance"] - e["clearance"]), self.clearance_tol))
        if self.use_learned_enable:
            checks.append(("enable", abs(l["enable"] - e["enable"]), self.enable_tol))

        bad = [name for name, val, tol in checks if val > tol]
        if bad:
            return False, ",".join(bad)
        return True, "accepted"

    def on_timer(self):
        self.seq += 1
        accept, reason = self.decide()

        e = self.empirical or {
            "seq": -1.0, "vx": 0.0, "yaw_rate": 0.0,
            "body_h": 0.32, "clearance": 0.045, "enable": 0.0,
        }
        l = self.learned or {
            "seq": -1.0, "vx": 0.0, "yaw_rate": 0.0,
            "body_h": 0.32, "clearance": 0.045, "enable": 0.0,
        }

        if accept:
            selected = dict(e)
            if self.use_learned_vx:
                selected["vx"] = l["vx"]
            if self.use_learned_yaw:
                selected["yaw_rate"] = l["yaw_rate"]
            if self.use_learned_body_h:
                selected["body_h"] = l["body_h"]
            if self.use_learned_clearance:
                selected["clearance"] = l["clearance"]
            if self.use_learned_enable:
                selected["enable"] = l["enable"]
            selected_source = "learned_selected_dims"
        else:
            selected = e
            selected_source = "empirical"

        msg = Float64MultiArray()
        msg.data = [
            float(self.seq),
            float(selected["vx"]),
            float(selected["yaw_rate"]),
            float(selected["body_h"]),
            float(selected["clearance"]),
            float(selected["enable"]),
        ]
        self.pub.publish(msg)

        self.writer.writerow({
            "t_wall": time.time(),
            "context": self.current_context,
            "gate_accept": accept,
            "reject_reason": reason,
            "emp_seq": e["seq"],
            "learned_seq": l["seq"],
            "selected_source": selected_source,
            "emp_vx": e["vx"],
            "learned_vx": l["vx"],
            "selected_vx": selected["vx"],
            "err_vx": l["vx"] - e["vx"],
            "emp_yaw_rate": e["yaw_rate"],
            "learned_yaw_rate": l["yaw_rate"],
            "selected_yaw_rate": selected["yaw_rate"],
            "err_yaw_rate": l["yaw_rate"] - e["yaw_rate"],
            "emp_body_h": e["body_h"],
            "learned_body_h": l["body_h"],
            "selected_body_h": selected["body_h"],
            "err_body_h": l["body_h"] - e["body_h"],
            "emp_clearance": e["clearance"],
            "learned_clearance": l["clearance"],
            "selected_clearance": selected["clearance"],
            "err_clearance": l["clearance"] - e["clearance"],
            "emp_enable": e["enable"],
            "learned_enable": l["enable"],
            "selected_enable": selected["enable"],
            "err_enable": l["enable"] - e["enable"],
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
    node = GatedSelectorDryRun()
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
