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


def env_list(name, default):
    raw = os.environ.get(name, "")
    if raw.strip() == "":
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def context_key(label):
    label = (label or "unknown").strip()
    known = {"flat", "start_flat", "rough", "upslope", "downslope", "goal_flat", "unknown"}
    return label if label in known else "unknown"


class J7GuardedMetaActionRefGate(Node):
    def __init__(self):
        super().__init__("tracer_phase_j7_guarded_meta_action_ref_gate_node_v0")

        self.emp_topic = os.environ.get("TRACER_PHASE_J7_EMPIRICAL_REF_TOPIC", "/tracer/empirical_mpc_reference")
        self.proj_topic = os.environ.get("TRACER_PHASE_J7_PROJECTED_REF_TOPIC", "/tracer/meta_action_projected_ref_shadow")
        self.theta_topic = os.environ.get("TRACER_PHASE_J7_THETA_TOPIC", "/tracer/meta_action_theta_shadow")
        self.context_topic = os.environ.get("TRACER_PHASE_J7_CONTEXT_TOPIC", "/tracer/terrain_context_label")
        self.odom_topic = os.environ.get("TRACER_PHASE_J7_ODOM_TOPIC", "/tracer/robot_odom_flat")

        # Shadow-only by default. Do NOT set this to /tracer/mpc_reference yet.
        self.out_topic = os.environ.get("TRACER_PHASE_J7_OUTPUT_REF_TOPIC", "/tracer/meta_action_guarded_ref_shadow")

        self.pub_hz = env_float("TRACER_PHASE_J7_PUB_HZ", 10.0)

        self.max_abs_delta_vx = env_float("TRACER_PHASE_J7_MAX_ABS_DELTA_VX", 0.030)
        self.max_abs_delta_yaw = env_float("TRACER_PHASE_J7_MAX_ABS_DELTA_YAW", 0.020)
        self.max_abs_delta_body_h = env_float("TRACER_PHASE_J7_MAX_ABS_DELTA_BODY_H", 0.006)
        self.max_abs_delta_clearance = env_float("TRACER_PHASE_J7_MAX_ABS_DELTA_CLEARANCE", 0.006)
        self.min_active_body_h = env_float("TRACER_PHASE_J7_MIN_ACTIVE_BODY_H", 0.314)
        self.goal_x_threshold = env_float("TRACER_PHASE_J7_GOAL_X_THRESHOLD", 7.90)
        self.max_input_age_s = env_float("TRACER_PHASE_J7_MAX_INPUT_AGE_S", 1.0)

        # Optional active-routing sequence alignment.
        #
        # J4 and J7 run on independent timers. Without this guard, J7 may
        # compare the current empirical reference against a projected
        # reference generated from the previous empirical sequence.
        #
        # Disabled by default to preserve historical shadow behavior.
        self.require_seq_match = (
            os.environ.get(
                "TRACER_PHASE_J7_REQUIRE_SEQ_MATCH",
                "0",
            )
            == "1"
        )

        # Active-mode safeguard. Disabled by default for shadow tests.
        # When enabled, stale/missing projected inputs output a conservative hold ref
        # instead of replaying a stale empirical ref.
        self.active_failsafe_hold = os.environ.get("TRACER_PHASE_J7_ACTIVE_FAILSAFE_HOLD", "0") == "1"
        self.failsafe_hold_vx = env_float("TRACER_PHASE_J7_FAILSAFE_HOLD_VX", 0.025)
        self.failsafe_body_h = env_float("TRACER_PHASE_J7_FAILSAFE_BODY_H", 0.320)
        self.failsafe_clearance = env_float("TRACER_PHASE_J7_FAILSAFE_CLEARANCE", 0.045)

        self.allowed_contexts = set(env_list("TRACER_PHASE_J7_ALLOWED_CONTEXTS", ["flat", "upslope", "downslope"]))

        self.ctx = "unknown"
        self.x = None
        self.y = None

        self.emp_ref = None
        self.emp_t = None

        self.proj_ref = None
        self.proj_t = None

        self.theta = None
        self.theta_t = None

        self.pub = self.create_publisher(Float64MultiArray, self.out_topic, 10)

        self.create_subscription(Float64MultiArray, self.emp_topic, self.on_emp, 10)
        self.create_subscription(Float64MultiArray, self.proj_topic, self.on_proj, 10)
        self.create_subscription(Float64MultiArray, self.theta_topic, self.on_theta, 10)
        self.create_subscription(String, self.context_topic, self.on_context, 10)
        self.create_subscription(Float64MultiArray, self.odom_topic, self.on_odom, 10)

        log_dir = Path(os.environ.get("TRACER_PHASE_J7_LOG_DIR", "logs/phase_j/j7_guarded_ref_gate_latest"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = log_dir / "guarded_meta_action_ref_gate_j7_v0.csv"
        self.log_f = open(self.log_csv, "w", newline="")
        self.writer = csv.DictWriter(
            self.log_f,
            fieldnames=[
                "t_wall", "context", "x", "y",
                "action_id", "hold_override",
                "emp_age_s", "proj_age_s", "theta_age_s",
                "emp_seq", "emp_vx", "emp_yaw_rate", "emp_body_h", "emp_clearance", "emp_enable",
                "proj_seq", "proj_vx", "proj_yaw_rate", "proj_body_h", "proj_clearance", "proj_enable",
                "out_seq", "out_vx", "out_yaw_rate", "out_body_h", "out_clearance", "out_enable",
                "delta_vx", "delta_yaw_rate", "delta_body_h", "delta_clearance",
                "j7_accept", "j7_reason", "j7_source",
            ],
        )
        self.writer.writeheader()
        self.log_f.flush()

        self.timer = self.create_timer(1.0 / max(self.pub_hz, 1e-6), self.on_timer)

        self.get_logger().info(f"[TRACER:J7] empirical input={self.emp_topic}")
        self.get_logger().info(f"[TRACER:J7] projected input={self.proj_topic}")
        self.get_logger().info(f"[TRACER:J7] theta input={self.theta_topic}")
        self.get_logger().info(f"[TRACER:J7] output={self.out_topic}")
        self.get_logger().info(f"[TRACER:J7] allowed_contexts={sorted(self.allowed_contexts)}")
        self.get_logger().info(f"[TRACER:J7] require_seq_match={self.require_seq_match}")
        self.get_logger().info(f"[TRACER:J7] active_failsafe_hold={self.active_failsafe_hold}")
        self.get_logger().info(f"[TRACER:J7] logging to {self.log_csv}")
        self.get_logger().info("[TRACER:J7] shadow-only unless output topic is explicitly changed")

    def on_context(self, msg):
        self.ctx = context_key(msg.data)

    def on_odom(self, msg):
        data = list(msg.data)
        if len(data) >= 2:
            self.x = float(data[0])
            self.y = float(data[1])

    def on_emp(self, msg):
        data = list(msg.data)
        if len(data) >= 6:
            self.emp_ref = [float(v) for v in data[:6]]
            self.emp_t = time.time()

    def on_proj(self, msg):
        data = list(msg.data)
        if len(data) >= 6:
            self.proj_ref = [float(v) for v in data[:6]]
            self.proj_t = time.time()

    def on_theta(self, msg):
        data = list(msg.data)
        if len(data) >= 9:
            self.theta = [float(v) for v in data[:9]]
            self.theta_t = time.time()

    def decide(self, now):
        if self.emp_ref is None:
            return False, "missing_empirical_ref"
        if self.proj_ref is None:
            return False, "missing_projected_ref"
        if self.theta is None:
            return False, "missing_theta"

        emp_age = now - self.emp_t if self.emp_t is not None else 999.0
        proj_age = now - self.proj_t if self.proj_t is not None else 999.0
        theta_age = now - self.theta_t if self.theta_t is not None else 999.0

        if emp_age > self.max_input_age_s:
            return False, "stale_empirical_ref"
        if proj_age > self.max_input_age_s:
            return False, "stale_projected_ref"
        if theta_age > self.max_input_age_s:
            return False, "stale_theta"

        ctx = context_key(self.ctx)
        action_id = int(round(self.theta[0]))
        hold_override = self.theta[8] >= 0.5

        if hold_override or action_id == 8 or ctx == "goal_flat":
            return False, "hold_or_goal_protected"
        if self.x is not None and self.x >= self.goal_x_threshold:
            return False, "goal_x_protected"

        emp = self.emp_ref
        proj = self.proj_ref

        if self.require_seq_match:
            emp_seq = int(round(emp[0]))
            proj_seq = int(round(proj[0]))

            if emp_seq != proj_seq:
                return False, "emp_proj_seq_mismatch"

        dvx = proj[1] - emp[1]
        dyaw = proj[2] - emp[2]
        dbh = proj[3] - emp[3]
        dcl = proj[4] - emp[4]

        if abs(dvx) > self.max_abs_delta_vx:
            return False, "delta_vx_too_large"
        if abs(dyaw) > self.max_abs_delta_yaw:
            return False, "delta_yaw_too_large"
        if abs(dbh) > self.max_abs_delta_body_h:
            return False, "delta_body_h_too_large"
        if abs(dcl) > self.max_abs_delta_clearance:
            return False, "delta_clearance_too_large"

        if proj[3] < self.min_active_body_h:
            return False, "body_h_too_low_for_active_candidate"

        if ctx not in self.allowed_contexts:
            return False, "context_not_allowed_for_first_active"

        return True, "accepted_guarded"

    def on_timer(self):
        if self.emp_ref is None:
            return

        now = time.time()
        accepted, reason = self.decide(now)

        stale_or_missing = reason in {
            "missing_projected_ref",
            "missing_theta",
            "stale_empirical_ref",
            "stale_projected_ref",
            "stale_theta",
        }

        if accepted and self.proj_ref is not None:
            out = list(self.proj_ref)
            source = "projected"
        elif self.active_failsafe_hold and stale_or_missing:
            seq = self.emp_ref[0] if self.emp_ref is not None else 0.0
            out = [
                seq,
                self.failsafe_hold_vx,
                0.0,
                self.failsafe_body_h,
                self.failsafe_clearance,
                1.0,
            ]
            source = "failsafe_hold"
        else:
            out = list(self.emp_ref)
            source = "empirical"

        msg = Float64MultiArray()
        msg.data = [float(v) for v in out]
        self.pub.publish(msg)

        emp = self.emp_ref
        proj = self.proj_ref if self.proj_ref is not None else [math.nan] * 6
        theta = self.theta if self.theta is not None else [math.nan] * 9

        emp_age = now - self.emp_t if self.emp_t is not None else math.nan
        proj_age = now - self.proj_t if self.proj_t is not None else math.nan
        theta_age = now - self.theta_t if self.theta_t is not None else math.nan

        self.writer.writerow({
            "t_wall": f"{now:.6f}",
            "context": context_key(self.ctx),
            "x": "" if self.x is None else f"{self.x:.6f}",
            "y": "" if self.y is None else f"{self.y:.6f}",
            "action_id": "" if math.isnan(theta[0]) else f"{theta[0]:.0f}",
            "hold_override": "" if math.isnan(theta[8]) else ("1" if theta[8] >= 0.5 else "0"),
            "emp_age_s": "" if math.isnan(emp_age) else f"{emp_age:.6f}",
            "proj_age_s": "" if math.isnan(proj_age) else f"{proj_age:.6f}",
            "theta_age_s": "" if math.isnan(theta_age) else f"{theta_age:.6f}",
            "emp_seq": f"{emp[0]:.0f}",
            "emp_vx": f"{emp[1]:.8f}",
            "emp_yaw_rate": f"{emp[2]:.8f}",
            "emp_body_h": f"{emp[3]:.8f}",
            "emp_clearance": f"{emp[4]:.8f}",
            "emp_enable": f"{emp[5]:.8f}",
            "proj_seq": "" if math.isnan(proj[0]) else f"{proj[0]:.0f}",
            "proj_vx": "" if math.isnan(proj[1]) else f"{proj[1]:.8f}",
            "proj_yaw_rate": "" if math.isnan(proj[2]) else f"{proj[2]:.8f}",
            "proj_body_h": "" if math.isnan(proj[3]) else f"{proj[3]:.8f}",
            "proj_clearance": "" if math.isnan(proj[4]) else f"{proj[4]:.8f}",
            "proj_enable": "" if math.isnan(proj[5]) else f"{proj[5]:.8f}",
            "out_seq": f"{out[0]:.0f}",
            "out_vx": f"{out[1]:.8f}",
            "out_yaw_rate": f"{out[2]:.8f}",
            "out_body_h": f"{out[3]:.8f}",
            "out_clearance": f"{out[4]:.8f}",
            "out_enable": f"{out[5]:.8f}",
            "delta_vx": "" if math.isnan(proj[1]) else f"{proj[1] - emp[1]:.8f}",
            "delta_yaw_rate": "" if math.isnan(proj[2]) else f"{proj[2] - emp[2]:.8f}",
            "delta_body_h": "" if math.isnan(proj[3]) else f"{proj[3] - emp[3]:.8f}",
            "delta_clearance": "" if math.isnan(proj[4]) else f"{proj[4] - emp[4]:.8f}",
            "j7_accept": "1" if accepted else "0",
            "j7_reason": reason,
            "j7_source": source,
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
    node = J7GuardedMetaActionRefGate()
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
