#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import socket
import time
from pathlib import Path
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def f(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    return f(os.environ.get(name, default), default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def terrain_onehot(name: str) -> List[float]:
    return [
        1.0 if name == "flat_normal" else 0.0,
        1.0 if name == "rough_mid" else 0.0,
        1.0 if name == "slope_5deg" else 0.0,
    ]


def beta_from_terrain(name: str) -> List[float]:
    if name == "flat_normal":
        return [0.55, 0.25, 0.20]
    if name == "rough_mid":
        return [0.30, 0.50, 0.20]
    if name == "slope_5deg":
        return [0.25, 0.55, 0.20]
    return [0.25, 0.55, 0.20]


def infer_rule_gms_label(terrain: str, vx: float, h: float, clr: float, enable: float) -> str:
    if enable < 0.5:
        return "disabled"

    # Current conservative command region from rollout logs:
    # rough/slope eventually become vx~0.035, h~0.35, clr~0.11.
    if vx <= 0.038 and h >= 0.345 and clr >= 0.100:
        return "conservative"

    if terrain == "flat_normal":
        return "fast"
    if terrain == "rough_mid":
        return "cautious_probe"
    if terrain == "slope_5deg":
        return "high_clearance_slow_probe"

    if clr >= 0.08:
        return "high_clearance_slow_probe"
    if vx <= 0.06:
        return "cautious_probe"
    return "fast"


def gate_proxy_from_rule_label(rule_label: str):
    if rule_label == "conservative":
        return {
            "gate_level_code": 2.0,
            "gate_action_code": 2.0,
            "gate_would_override": 1.0,
        }
    return {
        "gate_level_code": 0.0,
        "gate_action_code": 0.0,
        "gate_would_override": 0.0,
    }


def build_step_features(
    terrain: str,
    vx: float,
    yaw: float,
    h: float,
    clr: float,
    enable: float,
    odom_vx: float = 0.0,
) -> Dict[str, float]:
    beta = beta_from_terrain(terrain)
    rule_label = infer_rule_gms_label(terrain, vx, h, clr, enable)
    gate = gate_proxy_from_rule_label(rule_label)

    # These are shadow-mode proxy features. Once we subscribe to explicit RAM/GMS
    # debug topics, replace these proxies with true runtime gate values.
    #
    # Important:
    # The GMS classifier was trained from rollout CSV features where raw RAM risk
    # can be high even before the gate action switches to conservative.
    # Therefore, for slope high-clearance probe we keep gate_level/action low
    # but allow raw risk proxies to be high. This separates:
    #   - high_clearance_slow_probe: high raw risk, low gate action
    #   - conservative:             high raw risk, high gate action
    is_conservative_gate = gate["gate_level_code"] >= 2.0
    is_slope_probe = terrain == "slope_5deg" and rule_label == "high_clearance_slow_probe"

    if is_conservative_gate:
        gate_control_risk = 0.80
        gate_future_risk = 1.00
        gate_fallen_prob = 1.00
        gate_sigma_mean = 0.52
        gate_rho_norm = 5.0
    elif is_slope_probe:
        gate_control_risk = 0.80
        gate_future_risk = 1.00
        gate_fallen_prob = 1.00
        gate_sigma_mean = 0.52
        gate_rho_norm = 5.0
    else:
        gate_control_risk = 0.0
        gate_future_risk = 0.0
        gate_fallen_prob = 0.0
        gate_sigma_mean = 0.1
        gate_rho_norm = 0.0

    gate_recovery_prob = 0.0

    vx_scale = vx / 0.28 if 0.28 > 1e-6 else 1.0
    h_delta = h - 0.295
    clr_delta = clr - 0.030

    return {
        "mpc_vx": vx,
        "mpc_yaw": yaw,
        "mpc_body_height": h,
        "mpc_clearance": clr,
        "mpc_enable": enable,
        "beta_motion": beta[0],
        "beta_stability": beta[1],
        "beta_energy": beta[2],
        "gate_level_code": gate["gate_level_code"],
        "gate_action_code": gate["gate_action_code"],
        "gate_would_override": gate["gate_would_override"],
        "gate_control_risk": gate_control_risk,
        "gate_future_risk": gate_future_risk,
        "gate_fallen_prob": gate_fallen_prob,
        "gate_recovery_prob": gate_recovery_prob,
        "gate_sigma_mean": gate_sigma_mean,
        "gate_rho_norm": gate_rho_norm,
        "gate_vx_scale": vx_scale,
        "gate_h_delta": h_delta,
        "gate_clr_delta": clr_delta,
        "odom_vx": odom_vx,
    }


RAM_FEATURES = [
    "mpc_vx",
    "mpc_yaw",
    "mpc_body_height",
    "mpc_clearance",
    "mpc_enable",
    "beta_motion",
    "beta_stability",
    "beta_energy",
    "gate_level_code",
    "gate_action_code",
    "gate_would_override",
    "gate_control_risk",
    "gate_future_risk",
    "gate_fallen_prob",
    "gate_recovery_prob",
    "gate_sigma_mean",
    "gate_rho_norm",
    "gate_vx_scale",
    "gate_h_delta",
    "gate_clr_delta",
    "odom_vx",
]


def flatten_ram_window(window: List[Dict[str, float]], target_len: int = 30) -> List[float]:
    if not window:
        dummy = {k: 0.0 for k in RAM_FEATURES}
        window = [dummy]

    if len(window) < target_len:
        pad = [window[0]] * (target_len - len(window))
        use = pad + window
    else:
        use = window[-target_len:]

    x = []
    for step in use:
        x.extend([f(step.get(k, 0.0)) for k in RAM_FEATURES])
    return x


def build_gms_x(terrain: str, step: Dict[str, float]) -> List[float]:
    return terrain_onehot(terrain) + [
        step["beta_motion"],
        step["beta_stability"],
        step["beta_energy"],

        step["gate_level_code"],
        step["gate_action_code"],
        step["gate_would_override"],
        step["gate_control_risk"],
        step["gate_future_risk"],
        step["gate_fallen_prob"],
        step["gate_recovery_prob"],
        step["gate_sigma_mean"],
        step["gate_rho_norm"],
        step["gate_vx_scale"],
        step["gate_h_delta"],
        step["gate_clr_delta"],

        step["mpc_vx"],
        step["mpc_body_height"],
        step["mpc_clearance"],
        step["mpc_enable"],
    ]


def build_objective_x(terrain: str, step: Dict[str, float], elapsed: float, duration: float) -> List[float]:
    # Objective v1 was trained on episode-summary features.
    # In shadow mode v1, this is a runtime proxy, not a control-authoritative signal.
    duration_norm = min(1.0, max(0.0, duration / 30.0)) if duration > 0.0 else min(1.0, elapsed / 30.0)

    distance_proxy = max(0.0, min(1.5, step["mpc_vx"] * max(1.0, elapsed) / 40.0))

    return terrain_onehot(terrain) + [
        duration_norm,
        step["mpc_vx"],
        step["mpc_enable"],
        step["mpc_body_height"],
        step["mpc_clearance"],
        step["gate_fallen_prob"],
        step["gate_recovery_prob"],
        step["gate_level_code"],
        step["gate_action_code"],
        step["gate_would_override"],
        distance_proxy,
        1.0,
    ]


class LearnedStackShadowNode(Node):
    def __init__(self):
        super().__init__("tracer_learned_stack_shadow_node_v1")

        self.terrain = os.environ.get("TRACER_TERRAIN", os.environ.get("TRACER_SHADOW_TERRAIN", "flat_normal"))
        self.host = os.environ.get("TRACER_SUPERVISED_STACK_HOST", "127.0.0.1")
        self.port = env_int("TRACER_SUPERVISED_STACK_PORT", 50410)
        self.timeout_sec = env_float("TRACER_SUPERVISED_STACK_TIMEOUT_SEC", 0.20)
        self.hz = env_float("TRACER_SHADOW_HZ", 5.0)
        self.duration_sec = env_float("TRACER_SHADOW_DURATION_SEC", 0.0)
        self.window_len = env_int("TRACER_SHADOW_RAM_WINDOW", 30)

        default_out = f"data/shadow_logs/learned_stack_shadow_{self.terrain}_{int(time.time())}.csv"
        self.out_csv = Path(os.environ.get("TRACER_SHADOW_OUT_CSV", default_out))
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(self.timeout_sec)

        self.latest_mpc = None
        self.window: List[Dict[str, float]] = []
        self.t0 = time.time()
        self.n = 0

        self.sub = self.create_subscription(Float64MultiArray, "/tracer/mpc_reference", self.on_mpc, 10)

        self.fobj = self.out_csv.open("w", newline="")
        self.writer = csv.DictWriter(self.fobj, fieldnames=[
            "time_sec",
            "terrain",
            "rule_vx",
            "rule_yaw",
            "rule_body_height",
            "rule_clearance",
            "rule_enable",
            "rule_gms_label",
            "learned_ok",
            "learned_beta_motion",
            "learned_beta_stability",
            "learned_beta_energy",
            "learned_ram_intervention_score",
            "learned_ram_future_override_mean",
            "learned_gms_label",
            "learned_gms_prob",
            "gms_disagree",
            "error",
        ])
        self.writer.writeheader()
        self.fobj.flush()

        self.timer = self.create_timer(1.0 / max(0.1, self.hz), self.on_timer)

        self.get_logger().info(
            f"learned stack shadow started terrain={self.terrain} "
            f"udp={self.host}:{self.port} hz={self.hz} out={self.out_csv}"
        )

    def on_mpc(self, msg: Float64MultiArray):
        data = list(msg.data)
        if len(data) < 6:
            return

        self.latest_mpc = {
            "counter": f(data[0]),
            "vx": f(data[1]),
            "yaw": f(data[2]),
            "h": f(data[3]),
            "clr": f(data[4]),
            "enable": f(data[5]),
            "stamp": time.time(),
        }

    def query_udp(self, payload: dict) -> dict:
        self.sock.sendto(json.dumps(payload).encode("utf-8"), (self.host, self.port))
        data, _ = self.sock.recvfrom(65535)
        return json.loads(data.decode("utf-8"))

    def on_timer(self):
        elapsed = time.time() - self.t0

        if self.duration_sec > 0.0 and elapsed >= self.duration_sec:
            self.get_logger().info(f"shadow duration reached; closing csv={self.out_csv}")
            self.fobj.flush()
            self.fobj.close()
            os._exit(0)

        if self.latest_mpc is None:
            return

        m = self.latest_mpc
        step = build_step_features(
            self.terrain,
            m["vx"],
            m["yaw"],
            m["h"],
            m["clr"],
            m["enable"],
            odom_vx=0.0,
        )
        self.window.append(step)

        rule_label = infer_rule_gms_label(self.terrain, m["vx"], m["h"], m["clr"], m["enable"])

        payload = {
            "request_id": f"shadow_{self.terrain}_{self.n}",
            "terrain": self.terrain,
            "objective_x": build_objective_x(self.terrain, step, elapsed, self.duration_sec),
            "ram_x": flatten_ram_window(self.window, self.window_len),
            "gms_x": build_gms_x(self.terrain, step),
        }

        row = {
            "time_sec": elapsed,
            "terrain": self.terrain,
            "rule_vx": m["vx"],
            "rule_yaw": m["yaw"],
            "rule_body_height": m["h"],
            "rule_clearance": m["clr"],
            "rule_enable": m["enable"],
            "rule_gms_label": rule_label,
            "learned_ok": 0,
            "learned_beta_motion": "",
            "learned_beta_stability": "",
            "learned_beta_energy": "",
            "learned_ram_intervention_score": "",
            "learned_ram_future_override_mean": "",
            "learned_gms_label": "",
            "learned_gms_prob": "",
            "gms_disagree": "",
            "error": "",
        }

        try:
            resp = self.query_udp(payload)
            row["learned_ok"] = 1 if resp.get("ok", False) else 0

            obj = resp.get("objective") or {}
            if obj.get("ok", False):
                beta = obj.get("beta", [None, None, None])
                row["learned_beta_motion"] = beta[0]
                row["learned_beta_stability"] = beta[1]
                row["learned_beta_energy"] = beta[2]

            ram = resp.get("ram") or {}
            if ram.get("ok", False):
                row["learned_ram_intervention_score"] = ram.get("intervention_score", "")
                row["learned_ram_future_override_mean"] = ram.get("future_override_mean", "")

            gms = resp.get("gms") or {}
            if gms.get("ok", False):
                learned_label = gms.get("label", "")
                row["learned_gms_label"] = learned_label
                row["learned_gms_prob"] = gms.get("prob", "")
                row["gms_disagree"] = 1 if learned_label != rule_label else 0

            if not resp.get("ok", False):
                row["error"] = resp.get("error", "")

        except Exception as e:
            row["error"] = repr(e)

        self.writer.writerow(row)
        self.fobj.flush()
        self.n += 1

        if self.n % max(1, int(self.hz * 5.0)) == 0:
            self.get_logger().info(
                f"shadow n={self.n} rule={row['rule_gms_label']} "
                f"learned={row['learned_gms_label']} disagree={row['gms_disagree']} "
                f"ram={row['learned_ram_intervention_score']}"
            )


def main():
    rclpy.init()
    node = LearnedStackShadowNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
