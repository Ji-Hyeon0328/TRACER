#!/usr/bin/python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _safe_get(xs, i, default=0.0):
    try:
        return xs[i]
    except Exception:
        return default


class RamCalibrationRecorder(Node):
    def __init__(self):
        super().__init__("tracer_ram_calibration_recorder_v0")

        self.declare_parameter("terrain", os.environ.get("TRACER_TERRAIN", "unknown"))
        self.declare_parameter("duration_sec", float(os.environ.get("TRACER_RECORD_DURATION_SEC", "30.0")))
        self.declare_parameter("out_csv", os.environ.get("TRACER_RAM_CALIB_CSV", "data/ram_calibration_live/ram_calib_v0.csv"))
        self.declare_parameter("out_jsonl", os.environ.get("TRACER_RAM_CALIB_JSONL", "data/ram_calibration_live/ram_calib_v0.jsonl"))

        self.terrain = str(self.get_parameter("terrain").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.out_csv = Path(str(self.get_parameter("out_csv").value))
        self.out_jsonl = Path(str(self.get_parameter("out_jsonl").value))

        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

        self.t0 = time.time()
        self.last_mpc = None
        self.last_ram = None
        self.last_gate = None
        self.last_proprio = None

        self.rows = []

        self.create_subscription(Float64MultiArray, "/tracer/mpc_reference", self.on_mpc, 10)
        self.create_subscription(Float64MultiArray, "/tracer/ram_risk", self.on_ram, 10)
        self.create_subscription(Float64MultiArray, "/tracer/ram_gate_advice", self.on_gate, 10)
        self.create_subscription(Float64MultiArray, "/tracer/proprio_vector", self.on_proprio, 10)

        self.timer = self.create_timer(0.2, self.on_timer)

        self.get_logger().info(
            f"RAM calibration recorder started: terrain={self.terrain}, "
            f"duration={self.duration_sec:.1f}s, csv={self.out_csv}"
        )

    def on_mpc(self, msg: Float64MultiArray):
        self.last_mpc = list(msg.data)

    def on_ram(self, msg: Float64MultiArray):
        self.last_ram = list(msg.data)

    def on_gate(self, msg: Float64MultiArray):
        self.last_gate = list(msg.data)

    def on_proprio(self, msg: Float64MultiArray):
        self.last_proprio = list(msg.data)

    def make_row(self):
        now = time.time()
        elapsed = now - self.t0

        mpc = self.last_mpc or []
        ram = self.last_ram or []
        gate = self.last_gate or []
        proprio = self.last_proprio or []

        # /tracer/mpc_reference:
        # [counter, vx, yaw_rate, body_height, swing_clearance, enable]
        mpc_counter = _f(_safe_get(mpc, 0))
        mpc_vx = _f(_safe_get(mpc, 1))
        mpc_yaw = _f(_safe_get(mpc, 2))
        mpc_h = _f(_safe_get(mpc, 3))
        mpc_clr = _f(_safe_get(mpc, 4))
        mpc_enable = _f(_safe_get(mpc, 5))

        # /tracer/ram_risk current observed vector convention:
        # [future_risk, future_slip, future_invalid, run_valid, run_fallen,
        #  recovery_needed, sigma_mean, rho_norm, style_score, ctrl_ema, ...]
        ram_future_risk = _f(_safe_get(ram, 0))
        ram_future_slip = _f(_safe_get(ram, 1))
        ram_future_invalid = _f(_safe_get(ram, 2))
        ram_run_valid = _f(_safe_get(ram, 3))
        ram_run_fallen = _f(_safe_get(ram, 4))
        ram_recovery_needed = _f(_safe_get(ram, 5))
        ram_sigma_mean = _f(_safe_get(ram, 6))
        ram_rho_norm = _f(_safe_get(ram, 7))
        ram_style_score = _f(_safe_get(ram, 8))
        ram_ctrl_ema = _f(_safe_get(ram, 9))

        # /tracer/ram_gate_advice:
        # [mode_code, semantic_code, gate_level_code, action_code, would_override,
        #  control_risk, ctrl_ema, future_risk, fallen_prob, recovery_prob,
        #  sigma_mean, rho_norm, style_score, vx_scale, h_delta, clr_delta, ...]
        gate_level_code = _f(_safe_get(gate, 2))
        gate_action_code = _f(_safe_get(gate, 3))
        gate_would_override = _f(_safe_get(gate, 4))
        gate_control_risk = _f(_safe_get(gate, 5))
        gate_ctrl_ema = _f(_safe_get(gate, 6))
        gate_future_risk = _f(_safe_get(gate, 7))
        gate_fallen_prob = _f(_safe_get(gate, 8))
        gate_recovery_prob = _f(_safe_get(gate, 9))
        gate_sigma_mean = _f(_safe_get(gate, 10))
        gate_rho_norm = _f(_safe_get(gate, 11))
        gate_style_score = _f(_safe_get(gate, 12))
        gate_vx_scale = _f(_safe_get(gate, 13), 1.0)
        gate_h_delta = _f(_safe_get(gate, 14), 0.0)
        gate_clr_delta = _f(_safe_get(gate, 15), 0.0)

        # Basic proprio summary only. Full vector is not saved to keep csv compact.
        proprio_dim = len(proprio)
        proprio_abs_mean = 0.0
        if proprio:
            proprio_abs_mean = sum(abs(_f(v)) for v in proprio) / max(1, len(proprio))

        return {
            "wall_time": now,
            "elapsed": elapsed,
            "terrain": self.terrain,

            "mpc_counter": mpc_counter,
            "mpc_vx": mpc_vx,
            "mpc_yaw": mpc_yaw,
            "mpc_body_height": mpc_h,
            "mpc_clearance": mpc_clr,
            "mpc_enable": mpc_enable,

            "ram_future_risk": ram_future_risk,
            "ram_future_slip": ram_future_slip,
            "ram_future_invalid": ram_future_invalid,
            "ram_run_valid": ram_run_valid,
            "ram_run_fallen": ram_run_fallen,
            "ram_recovery_needed": ram_recovery_needed,
            "ram_sigma_mean": ram_sigma_mean,
            "ram_rho_norm": ram_rho_norm,
            "ram_style_score": ram_style_score,
            "ram_ctrl_ema": ram_ctrl_ema,

            "gate_level_code": gate_level_code,
            "gate_action_code": gate_action_code,
            "gate_would_override": gate_would_override,
            "gate_control_risk": gate_control_risk,
            "gate_ctrl_ema": gate_ctrl_ema,
            "gate_future_risk": gate_future_risk,
            "gate_fallen_prob": gate_fallen_prob,
            "gate_recovery_prob": gate_recovery_prob,
            "gate_sigma_mean": gate_sigma_mean,
            "gate_rho_norm": gate_rho_norm,
            "gate_style_score": gate_style_score,
            "gate_vx_scale": gate_vx_scale,
            "gate_h_delta": gate_h_delta,
            "gate_clr_delta": gate_clr_delta,

            "proprio_dim": proprio_dim,
            "proprio_abs_mean": proprio_abs_mean,
        }

    def flush(self):
        if not self.rows:
            return

        fields = list(self.rows[0].keys())

        with self.out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

        with self.out_jsonl.open("w") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")

        self.get_logger().info(
            f"wrote {len(self.rows)} rows: csv={self.out_csv}, jsonl={self.out_jsonl}"
        )

    def on_timer(self):
        elapsed = time.time() - self.t0
        self.rows.append(self.make_row())

        if elapsed >= self.duration_sec:
            self.flush()
            self.get_logger().info("finished RAM calibration recording")
            # rclpy.shutdown() can occasionally leave the process hanging
            # when called from a timer callback. The data is already flushed,
            # so force a clean process exit for shell sweep scripts.
            try:
                rclpy.shutdown()
            except Exception:
                pass
            os._exit(0)


def main():
    rclpy.init()
    node = RamCalibrationRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.flush()
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass


if __name__ == "__main__":
    main()
