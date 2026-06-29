#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    # .../TRACER/ros2_ws/src/tracer_a1_qpmc_adapter/scripts/file.py
    return here.parents[4]


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracer_core.highlevel.ram_scalar_v3_runtime_bridge import (  # noqa: E402
    RAMScalarV3RuntimeBridge,
    build_ram_scalar_v3_row,
)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _now() -> float:
    return time.time()


class TracerRAMScalarV3ShadowNode(Node):
    def __init__(self):
        super().__init__("tracer_ram_scalar_v3_shadow_node")

        self.declare_parameter("ram_host", os.environ.get("TRACER_RAM_SCALAR_V3_HOST", "127.0.0.1"))
        self.declare_parameter("ram_port", int(os.environ.get("TRACER_RAM_SCALAR_V3_PORT", "50230")))
        self.declare_parameter("ram_timeout_sec", float(os.environ.get("TRACER_RAM_SCALAR_V3_TIMEOUT_SEC", "0.05")))
        self.declare_parameter("hz", float(os.environ.get("TRACER_RAM_SCALAR_V3_SHADOW_HZ", "10.0")))

        self.declare_parameter("mpc_topic", os.environ.get("TRACER_RAM_SCALAR_V3_MPC_TOPIC", "/tracer/mpc_reference"))
        self.declare_parameter("beta_topic", os.environ.get("TRACER_RAM_SCALAR_V3_BETA_TOPIC", "/tracer/objective_weights"))
        self.declare_parameter("proprio_topic", os.environ.get("TRACER_RAM_SCALAR_V3_PROPRIO_TOPIC", "/tracer/proprio_flat"))
        self.declare_parameter("odom_topic", os.environ.get("TRACER_RAM_SCALAR_V3_ODOM_TOPIC", "/tracer/robot_odom_flat"))

        self.ram_host = str(self.get_parameter("ram_host").value)
        self.ram_port = int(self.get_parameter("ram_port").value)
        self.ram_timeout_sec = float(self.get_parameter("ram_timeout_sec").value)
        self.hz = float(self.get_parameter("hz").value)

        self.mpc_topic = str(self.get_parameter("mpc_topic").value)
        self.beta_topic = str(self.get_parameter("beta_topic").value)
        self.proprio_topic = str(self.get_parameter("proprio_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)

        self.bridge = RAMScalarV3RuntimeBridge(
            host=self.ram_host,
            port=self.ram_port,
            timeout_sec=self.ram_timeout_sec,
            fail_safe_risk=1.0,
        )

        self.latest_mpc = None
        self.latest_beta = None
        self.latest_proprio = None
        self.latest_odom = None

        self.latest_mpc_wall = None
        self.latest_beta_wall = None
        self.latest_proprio_wall = None
        self.latest_odom_wall = None

        self.create_subscription(Float64MultiArray, self.mpc_topic, self.on_mpc, 10)
        self.create_subscription(Float64MultiArray, self.beta_topic, self.on_beta, 10)
        self.create_subscription(Float64MultiArray, self.proprio_topic, self.on_proprio, 10)
        self.create_subscription(Float64MultiArray, self.odom_topic, self.on_odom, 10)

        self.shadow_pub = self.create_publisher(String, "/tracer/ram_scalar_v3_shadow", 10)
        self.risk_pub = self.create_publisher(Float64MultiArray, "/tracer/ram_scalar_v3_risk", 10)

        self.timer = self.create_timer(1.0 / max(1e-6, self.hz), self.on_timer)

        self.get_logger().info(
            "RAM scalar v3 shadow node started "
            f"server={self.ram_host}:{self.ram_port} "
            f"mpc={self.mpc_topic} beta={self.beta_topic} "
            f"proprio={self.proprio_topic} odom={self.odom_topic}"
        )

    def on_mpc(self, msg: Float64MultiArray) -> None:
        self.latest_mpc = list(msg.data)
        self.latest_mpc_wall = _now()

    def on_beta(self, msg: Float64MultiArray) -> None:
        self.latest_beta = list(msg.data)
        self.latest_beta_wall = _now()

    def on_proprio(self, msg: Float64MultiArray) -> None:
        self.latest_proprio = list(msg.data)
        self.latest_proprio_wall = _now()

    def on_odom(self, msg: Float64MultiArray) -> None:
        self.latest_odom = list(msg.data)
        self.latest_odom_wall = _now()

    def _age(self, stamp: float | None) -> float:
        if stamp is None:
            return 9999.0
        return max(0.0, _now() - float(stamp))

    def _build_row(self) -> dict[str, float]:
        # /tracer/mpc_reference layout:
        # [counter, vx, yaw_rate, body_height, swing_clearance, enable]
        mpc = self.latest_mpc or []
        beta = self.latest_beta or []

        # We keep proprio/odom layouts intentionally permissive.
        # Known useful convention:
        # proprio_flat may begin with [base_x, base_y, base_z, ...]
        # odom_flat may begin with [x, y, z, vx, ...]
        proprio = self.latest_proprio or []
        odom = self.latest_odom or []

        proprio_arr = np.asarray(proprio, dtype=float) if len(proprio) else np.asarray([], dtype=float)
        if proprio_arr.size:
            finite = proprio_arr[np.isfinite(proprio_arr)]
            prop_abs_mean = float(np.mean(np.abs(finite))) if finite.size else 0.0
            prop_abs_max = float(np.max(np.abs(finite))) if finite.size else 0.0
        else:
            prop_abs_mean = 0.0
            prop_abs_max = 0.0

        row = build_ram_scalar_v3_row(
            mpc_vx=_as_float(mpc[1], 0.0) if len(mpc) > 1 else 0.0,
            mpc_yaw=_as_float(mpc[2], 0.0) if len(mpc) > 2 else 0.0,
            mpc_body_height=_as_float(mpc[3], 0.0) if len(mpc) > 3 else 0.0,
            mpc_clearance=_as_float(mpc[4], 0.0) if len(mpc) > 4 else 0.0,
            mpc_enable=_as_float(mpc[5], 0.0) if len(mpc) > 5 else 0.0,

            beta_motion=_as_float(beta[0], 0.0) if len(beta) > 0 else 0.0,
            beta_stability=_as_float(beta[1], 0.0) if len(beta) > 1 else 0.0,
            beta_energy=_as_float(beta[2], 0.0) if len(beta) > 2 else 0.0,

            proprio_abs_mean=prop_abs_mean,
            proprio_abs_max=prop_abs_max,
            proprio_base_x=_as_float(proprio[0], 0.0) if len(proprio) > 0 else 0.0,
            proprio_base_y=_as_float(proprio[1], 0.0) if len(proprio) > 1 else 0.0,
            proprio_base_z=_as_float(proprio[2], 0.0) if len(proprio) > 2 else 0.0,

            odom_x=_as_float(odom[0], 0.0) if len(odom) > 0 else 0.0,
            odom_y=_as_float(odom[1], 0.0) if len(odom) > 1 else 0.0,
            odom_z=_as_float(odom[2], 0.0) if len(odom) > 2 else 0.0,
            # The scalar RAM v3 training dataset had non-finite odom_vx values
            # that were sanitized during training. Keep odom_vx disabled in this
            # shadow node until the recorder/dataset stores reliable velocity.
            odom_vx=0.0,

            age_mpc=self._age(self.latest_mpc_wall),
            age_beta=self._age(self.latest_beta_wall),
            age_proprio=self._age(self.latest_proprio_wall),
            age_odom=self._age(self.latest_odom_wall),
            age_debug=0.0,
        )
        return row

    def _input_fresh(self, max_age_sec: float = 1.0) -> tuple[bool, list[str]]:
        missing_or_stale: list[str] = []
        checks = [
            ("mpc", self.latest_mpc_wall),
            ("beta", self.latest_beta_wall),
            ("proprio", self.latest_proprio_wall),
            ("odom", self.latest_odom_wall),
        ]
        for name, stamp in checks:
            age = self._age(stamp)
            if age > max_age_sec:
                missing_or_stale.append(name)
        return (len(missing_or_stale) == 0), missing_or_stale

    def on_timer(self):
        row = self._build_row()
        input_fresh, missing_or_stale = self._input_fresh(max_age_sec=1.0)

        if not input_fresh:
            payload = {
                "stamp_wall": time.time(),
                "ok": False,
                "risk": 1.0,
                "bad_prob": 1.0,
                "good_prob": 0.0,
                "probabilities": {
                    "future_fallen_height": 1.0,
                    "future_low_height": 1.0,
                    "future_low_progress": 1.0,
                    "future_bad_locomotion": 1.0,
                    "future_good_locomotion": 0.0,
                },
                "row": row,
                "age_sec": 0.0,
                "error": "stale_or_missing_inputs:" + ",".join(missing_or_stale),
            }

            s = String()
            s.data = json.dumps(payload)
            self.shadow_pub.publish(s)

            arr = Float64MultiArray()
            arr.data = [1.0, 1.0, 0.0, 0.0]
            self.risk_pub.publish(arr)
            return

        out = self.bridge.query_row(row)

        payload = {
            "stamp_wall": _now(),
            "ok": out.ok,
            "risk": out.risk,
            "bad_prob": out.bad_prob,
            "good_prob": out.good_prob,
            "probabilities": out.probabilities,
            "age_sec": out.age_sec,
            "error": out.error,
            "row": row,
            "topics": {
                "mpc": self.mpc_topic,
                "beta": self.beta_topic,
                "proprio": self.proprio_topic,
                "odom": self.odom_topic,
            },
        }

        s = String()
        s.data = json.dumps(payload)
        self.shadow_pub.publish(s)

        arr = Float64MultiArray()
        arr.data = [
            float(out.risk),
            float(out.bad_prob),
            float(out.good_prob),
            1.0 if out.ok else 0.0,
        ]
        self.risk_pub.publish(arr)


def main():
    rclpy.init()
    node = TracerRAMScalarV3ShadowNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
