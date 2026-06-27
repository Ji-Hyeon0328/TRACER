#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


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


def get_path(obj: Any, path: str, default=None):
    cur = obj
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def parse_beta(policy_input: Dict[str, Any], terrain: str) -> List[float]:
    b = policy_input.get("beta", None)

    if isinstance(b, list) and len(b) >= 3:
        return [f(b[0]), f(b[1]), f(b[2])]

    if isinstance(b, dict):
        # Common possible names.
        candidates = [
            ("motion", "stability", "energy"),
            ("beta_motion", "beta_stability", "beta_energy"),
            ("motion_weight", "stability_weight", "energy_weight"),
            ("b_motion", "b_stability", "b_energy"),
        ]
        for a, c, d in candidates:
            if a in b and c in b and d in b:
                return [f(b[a]), f(b[c]), f(b[d])]

        # Dataclass/jsonable ObjectiveWeights may use these names.
        if "beta_v" in b and "beta_s" in b and "beta_e" in b:
            return [f(b["beta_v"]), f(b["beta_s"]), f(b["beta_e"])]

    return beta_from_terrain(terrain)


def level_code(x: Any) -> float:
    if isinstance(x, (int, float)):
        return f(x)

    s = str(x).lower()
    if s in ("", "none", "unknown", "safe", "normal", "validated", "valid", "low"):
        return 0.0
    if "caution" in s or "warn" in s or "probe" in s:
        return 1.0
    if "unstable" in s or "danger" in s or "conservative" in s or "stop" in s:
        return 2.0
    return 0.0


def action_code(x: Any) -> float:
    if isinstance(x, (int, float)):
        return f(x)

    s = str(x).lower()
    if s in ("", "none", "unknown", "no_override", "pass", "allow"):
        return 0.0
    if "probe" in s or "caution" in s or "slow" in s:
        return 1.0
    if "conservative" in s or "override" in s or "disable" in s or "stop" in s:
        return 2.0
    return 0.0


def rho_norm_from_policy_input(policy_input: Dict[str, Any]) -> float:
    ram = policy_input.get("ram", {})
    if not isinstance(ram, dict):
        return 0.0

    for key in ("rho", "rho_vec", "rho_latent", "mismatch_latent"):
        v = ram.get(key, None)
        if isinstance(v, list) and v:
            return math.sqrt(sum(f(x) * f(x) for x in v))

    for key in ("rho_norm", "mismatch_norm"):
        if key in ram:
            return f(ram[key])

    return 0.0


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
        use = [window[0]] * (target_len - len(window)) + window
    else:
        use = window[-target_len:]

    x = []
    for step in use:
        x.extend([f(step.get(k, 0.0)) for k in RAM_FEATURES])
    return x


def normalize_label(x: Any) -> str:
    s = str(x or "").strip()
    aliases = {
        "validated_locomotion": "fast",
        "validated_fast": "fast",
        "locomotion": "fast",
        "high_clearance": "high_clearance_slow_probe",
        "slow_probe": "high_clearance_slow_probe",
        "cautious": "cautious_probe",
    }
    return aliases.get(s, s)


def build_step_from_debug(dbg: Dict[str, Any]) -> Dict[str, float]:
    terrain = str(dbg.get("terrain", "unknown"))
    final_command = dbg.get("final_command") or {}
    gms_in = dbg.get("gms_in") or {}
    gms_out = dbg.get("gms_out") or {}
    policy_input = dbg.get("policy_input") or {}

    beta = parse_beta(policy_input, terrain)

    vx = f(final_command.get("vx", 0.0))
    yaw = f(final_command.get("yaw_rate", 0.0))
    h = f(final_command.get("body_height", 0.0))
    clr = f(final_command.get("swing_clearance", 0.0))
    enable = f(final_command.get("enable", 1.0))

    gl = level_code(gms_in.get("ram_level", "unknown"))
    ga = action_code(gms_in.get("ram_gate_action", "unknown"))

    control_risk = f(gms_in.get("control_risk", 0.0))
    fallen_prob = f(gms_in.get("fallen_prob", 0.0))
    recovery_prob = f(gms_in.get("recovery_prob", 0.0))
    sigma_mean = f(gms_in.get("sigma_mean", 0.0))

    future_risk = max(control_risk, fallen_prob, recovery_prob)
    would_override = 1.0 if ga > 0.0 else 0.0

    odom_vx = f(get_path(policy_input, "robot_state.odom_vx", 0.0))

    return {
        "mpc_vx": vx,
        "mpc_yaw": yaw,
        "mpc_body_height": h,
        "mpc_clearance": clr,
        "mpc_enable": enable,
        "beta_motion": beta[0],
        "beta_stability": beta[1],
        "beta_energy": beta[2],
        "gate_level_code": gl,
        "gate_action_code": ga,
        "gate_would_override": would_override,
        "gate_control_risk": control_risk,
        "gate_future_risk": future_risk,
        "gate_fallen_prob": fallen_prob,
        "gate_recovery_prob": recovery_prob,
        "gate_sigma_mean": sigma_mean,
        "gate_rho_norm": rho_norm_from_policy_input(policy_input),
        "gate_vx_scale": f(gms_out.get("vx_scale", 1.0)),
        "gate_h_delta": f(gms_out.get("body_height_delta", h - 0.295)),
        "gate_clr_delta": f(gms_out.get("clearance_delta", clr - 0.030)),
        "odom_vx": odom_vx,
    }


def build_gms_x(dbg: Dict[str, Any], step: Dict[str, float]) -> List[float]:
    terrain = str(dbg.get("terrain", "unknown"))
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


def build_objective_x(dbg: Dict[str, Any], step: Dict[str, float]) -> List[float]:
    terrain = str(dbg.get("terrain", "unknown"))
    elapsed = f(dbg.get("elapsed_sec", 0.0))
    duration_norm = min(1.0, max(0.0, elapsed / 30.0))
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


class LearnedStackShadowNodeV2(Node):
    def __init__(self):
        super().__init__("tracer_learned_stack_shadow_node_v2")

        self.host = os.environ.get("TRACER_SUPERVISED_STACK_HOST", "127.0.0.1")
        self.port = env_int("TRACER_SUPERVISED_STACK_PORT", 50410)
        self.timeout_sec = env_float("TRACER_SUPERVISED_STACK_TIMEOUT_SEC", 0.20)
        self.duration_sec = env_float("TRACER_SHADOW_DURATION_SEC", 0.0)
        self.window_len = env_int("TRACER_SHADOW_RAM_WINDOW", 30)

        terrain = os.environ.get("TRACER_TERRAIN", os.environ.get("TRACER_SHADOW_TERRAIN", "unknown"))
        default_out = f"data/shadow_logs_v2/learned_stack_shadow_v2_{terrain}_{int(time.time())}.csv"
        self.out_csv = Path(os.environ.get("TRACER_SHADOW_OUT_CSV", default_out))
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(self.timeout_sec)

        self.window: List[Dict[str, float]] = []
        self.t0 = time.time()
        self.n = 0

        self.fobj = self.out_csv.open("w", newline="")
        self.writer = csv.DictWriter(self.fobj, fieldnames=[
            "time_sec",
            "counter",
            "terrain",

            "rule_gms_label",
            "rule_gms_reason",
            "rule_ram_level",
            "rule_ram_gate_action",
            "rule_control_risk",
            "rule_fallen_prob",
            "rule_recovery_prob",
            "rule_sigma_mean",

            "rule_beta_motion",
            "rule_beta_stability",
            "rule_beta_energy",

            "final_vx",
            "final_body_height",
            "final_clearance",
            "final_enable",

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

        self.sub = self.create_subscription(
            String,
            "/tracer/highlevel_debug",
            self.on_debug,
            10,
        )

        self.timer = self.create_timer(0.5, self.on_timer)

        self.get_logger().info(
            f"learned stack shadow v2 started udp={self.host}:{self.port} "
            f"out={self.out_csv} duration={self.duration_sec}"
        )

    def on_timer(self):
        if self.duration_sec > 0.0 and (time.time() - self.t0) >= self.duration_sec:
            self.get_logger().info(f"shadow v2 duration reached; closing csv={self.out_csv}")
            self.fobj.flush()
            self.fobj.close()
            os._exit(0)

    def query_udp(self, payload: dict) -> dict:
        self.sock.sendto(json.dumps(payload).encode("utf-8"), (self.host, self.port))
        data, _ = self.sock.recvfrom(65535)
        return json.loads(data.decode("utf-8"))

    def on_debug(self, msg: String):
        elapsed = time.time() - self.t0

        try:
            dbg = json.loads(msg.data)
        except Exception as e:
            self.writer.writerow({
                "time_sec": elapsed,
                "counter": "",
                "terrain": "",
                "learned_ok": 0,
                "error": f"bad_debug_json: {e!r}",
            })
            self.fobj.flush()
            return

        terrain = str(dbg.get("terrain", "unknown"))
        gms_in = dbg.get("gms_in") or {}
        gms_out = dbg.get("gms_out") or {}
        policy_input = dbg.get("policy_input") or {}
        final_command = dbg.get("final_command") or {}

        step = build_step_from_debug(dbg)
        self.window.append(step)

        rule_label = normalize_label(gms_out.get("mode", ""))
        beta = parse_beta(policy_input, terrain)

        payload = {
            "request_id": f"shadow_v2_{terrain}_{self.n}",
            "terrain": terrain,
            "objective_x": build_objective_x(dbg, step),
            "ram_x": flatten_ram_window(self.window, self.window_len),
            "gms_x": build_gms_x(dbg, step),
        }

        row = {
            "time_sec": elapsed,
            "counter": dbg.get("counter", ""),
            "terrain": terrain,

            "rule_gms_label": rule_label,
            "rule_gms_reason": gms_out.get("reason", ""),
            "rule_ram_level": gms_in.get("ram_level", ""),
            "rule_ram_gate_action": gms_in.get("ram_gate_action", ""),
            "rule_control_risk": gms_in.get("control_risk", ""),
            "rule_fallen_prob": gms_in.get("fallen_prob", ""),
            "rule_recovery_prob": gms_in.get("recovery_prob", ""),
            "rule_sigma_mean": gms_in.get("sigma_mean", ""),

            "rule_beta_motion": beta[0],
            "rule_beta_stability": beta[1],
            "rule_beta_energy": beta[2],

            "final_vx": final_command.get("vx", ""),
            "final_body_height": final_command.get("body_height", ""),
            "final_clearance": final_command.get("swing_clearance", ""),
            "final_enable": final_command.get("enable", ""),

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
                lb = obj.get("beta", [None, None, None])
                row["learned_beta_motion"] = lb[0]
                row["learned_beta_stability"] = lb[1]
                row["learned_beta_energy"] = lb[2]

            ram = resp.get("ram") or {}
            if ram.get("ok", False):
                row["learned_ram_intervention_score"] = ram.get("intervention_score", "")
                row["learned_ram_future_override_mean"] = ram.get("future_override_mean", "")

            gms = resp.get("gms") or {}
            if gms.get("ok", False):
                learned_label = normalize_label(gms.get("label", ""))
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
        if self.n % 25 == 0:
            self.get_logger().info(
                f"shadow_v2 n={self.n} terrain={terrain} "
                f"rule={row['rule_gms_label']} learned={row['learned_gms_label']} "
                f"disagree={row['gms_disagree']} ram={row['learned_ram_intervention_score']}"
            )


def main():
    rclpy.init()
    node = LearnedStackShadowNodeV2()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
