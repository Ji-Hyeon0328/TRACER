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


def f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def get(xs: list[float] | None, i: int, default: float = 0.0) -> float:
    if xs is None:
        return default
    try:
        return f(xs[i], default)
    except Exception:
        return default


def mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    i = int(round((len(ys) - 1) * q))
    i = max(0, min(len(ys) - 1, i))
    return ys[i]


class RolloutEpisodeRecorder(Node):
    def __init__(self):
        super().__init__("tracer_rollout_episode_recorder_v0")

        self.declare_parameter("terrain", os.environ.get("TRACER_TERRAIN", "unknown"))
        self.declare_parameter("policy_id", os.environ.get("TRACER_POLICY_ID", "unknown_policy"))
        self.declare_parameter("episode_id", os.environ.get("TRACER_EPISODE_ID", f"ep_{int(time.time())}"))
        self.declare_parameter("duration_sec", float(os.environ.get("TRACER_ROLLOUT_DURATION_SEC", "30.0")))
        self.declare_parameter("sample_hz", float(os.environ.get("TRACER_ROLLOUT_SAMPLE_HZ", "5.0")))
        self.declare_parameter("out_dir", os.environ.get("TRACER_ROLLOUT_OUT_DIR", "data/rollout_dataset_v0"))

        self.terrain = str(self.get_parameter("terrain").value)
        self.policy_id = str(self.get_parameter("policy_id").value)
        self.episode_id = str(self.get_parameter("episode_id").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.sample_hz = float(self.get_parameter("sample_hz").value)
        self.out_dir = Path(str(self.get_parameter("out_dir").value))

        self.step_dir = self.out_dir / "episodes"
        self.summary_dir = self.out_dir / "summaries"
        self.step_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        self.step_csv = self.step_dir / f"{self.episode_id}.csv"
        self.summary_json = self.summary_dir / f"{self.episode_id}.json"

        self.t0 = time.time()
        self.rows: list[dict[str, Any]] = []

        self.latest: dict[str, tuple[float, list[float]]] = {}

        self.create_subscription(Float64MultiArray, "/tracer/mpc_reference", self.cb("mpc"), 10)
        self.create_subscription(Float64MultiArray, "/tracer/objective_weights", self.cb("beta"), 10)
        self.create_subscription(Float64MultiArray, "/tracer/ram_risk", self.cb("ram"), 10)
        self.create_subscription(Float64MultiArray, "/tracer/ram_gate_advice", self.cb("gate"), 10)
        self.create_subscription(Float64MultiArray, "/tracer/proprio_vector", self.cb("proprio"), 10)
        self.create_subscription(Float64MultiArray, "/tracer/robot_odom_flat", self.cb("odom"), 10)

        period = 1.0 / max(0.5, self.sample_hz)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            f"rollout recorder started episode={self.episode_id} terrain={self.terrain} "
            f"policy={self.policy_id} duration={self.duration_sec:.1f}s "
            f"csv={self.step_csv}"
        )

    def cb(self, name: str):
        def _inner(msg: Float64MultiArray):
            self.latest[name] = (time.time(), list(msg.data))
        return _inner

    def arr(self, name: str) -> list[float] | None:
        return self.latest.get(name, (0.0, None))[1]

    def age(self, name: str, now: float) -> float:
        t = self.latest.get(name, (0.0, []))[0]
        if t <= 0.0:
            return 9999.0
        return now - t

    def make_row(self) -> dict[str, Any]:
        now = time.time()
        elapsed = now - self.t0

        mpc = self.arr("mpc")
        beta = self.arr("beta")
        ram = self.arr("ram")
        gate = self.arr("gate")
        proprio = self.arr("proprio")
        odom = self.arr("odom")

        proprio_abs_mean = 0.0
        proprio_abs_max = 0.0
        if proprio:
            vals = [abs(f(v)) for v in proprio]
            proprio_abs_mean = mean(vals)
            proprio_abs_max = max(vals) if vals else 0.0

        # /tracer/mpc_reference:
        # [counter, vx, yaw_rate, body_height, clearance, enable]
        mpc_counter = get(mpc, 0)
        mpc_vx = get(mpc, 1)
        mpc_yaw = get(mpc, 2)
        mpc_h = get(mpc, 3)
        mpc_clr = get(mpc, 4)
        mpc_enable = get(mpc, 5)

        # /tracer/objective_weights:
        # [beta_motion, beta_stability, beta_energy]
        beta_motion = get(beta, 0)
        beta_stability = get(beta, 1)
        beta_energy = get(beta, 2)

        # /tracer/ram_risk observed convention:
        # [future_risk, future_slip, future_invalid, run_valid, run_fallen,
        #  recovery_needed, sigma_mean, rho_norm, style_score, ctrl_ema, ...]
        ram_future_risk = get(ram, 0)
        ram_future_slip = get(ram, 1)
        ram_future_invalid = get(ram, 2)
        ram_run_valid = get(ram, 3)
        ram_run_fallen = get(ram, 4)
        ram_recovery_needed = get(ram, 5)
        ram_sigma_mean = get(ram, 6)
        ram_rho_norm = get(ram, 7)
        ram_ctrl_ema = get(ram, 9)

        # /tracer/ram_gate_advice:
        # [mode_code, semantic_code, gate_level_code, action_code, would_override,
        #  control_risk, ctrl_ema, future_risk, fallen_prob, recovery_prob,
        #  sigma_mean, rho_norm, style_score, vx_scale, h_delta, clr_delta, ...]
        gate_level_code = get(gate, 2)
        gate_action_code = get(gate, 3)
        gate_would_override = get(gate, 4)
        gate_control_risk = get(gate, 5)
        gate_ctrl_ema = get(gate, 6)
        gate_future_risk = get(gate, 7)
        gate_fallen_prob = get(gate, 8)
        gate_recovery_prob = get(gate, 9)
        gate_sigma_mean = get(gate, 10)
        gate_rho_norm = get(gate, 11)
        gate_vx_scale = get(gate, 13, 1.0)
        gate_h_delta = get(gate, 14)
        gate_clr_delta = get(gate, 15)

        odom_x = get(odom, 0, float("nan"))
        odom_y = get(odom, 1, float("nan"))
        odom_z = get(odom, 2, float("nan"))
        odom_vx = get(odom, 3, float("nan"))

        return {
            "wall_time": now,
            "elapsed": elapsed,
            "episode_id": self.episode_id,
            "terrain": self.terrain,
            "policy_id": self.policy_id,

            "mpc_counter": mpc_counter,
            "mpc_vx": mpc_vx,
            "mpc_yaw": mpc_yaw,
            "mpc_body_height": mpc_h,
            "mpc_clearance": mpc_clr,
            "mpc_enable": mpc_enable,

            "beta_motion": beta_motion,
            "beta_stability": beta_stability,
            "beta_energy": beta_energy,

            "ram_future_risk": ram_future_risk,
            "ram_future_slip": ram_future_slip,
            "ram_future_invalid": ram_future_invalid,
            "ram_run_valid": ram_run_valid,
            "ram_run_fallen": ram_run_fallen,
            "ram_recovery_needed": ram_recovery_needed,
            "ram_sigma_mean": ram_sigma_mean,
            "ram_rho_norm": ram_rho_norm,
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
            "gate_vx_scale": gate_vx_scale,
            "gate_h_delta": gate_h_delta,
            "gate_clr_delta": gate_clr_delta,

            "proprio_dim": len(proprio or []),
            "proprio_abs_mean": proprio_abs_mean,
            "proprio_abs_max": proprio_abs_max,

            "odom_dim": len(odom or []),
            "odom_x": odom_x,
            "odom_y": odom_y,
            "odom_z": odom_z,
            "odom_vx": odom_vx,

            "age_mpc": self.age("mpc", now),
            "age_beta": self.age("beta", now),
            "age_ram": self.age("ram", now),
            "age_gate": self.age("gate", now),
            "age_proprio": self.age("proprio", now),
            "age_odom": self.age("odom", now),
        }

    def write_outputs(self):
        if not self.rows:
            return

        fields = list(self.rows[0].keys())
        with self.step_csv.open("w", newline="") as fobj:
            w = csv.DictWriter(fobj, fieldnames=fields)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

        summary = self.make_summary()
        with self.summary_json.open("w") as fobj:
            json.dump(summary, fobj, indent=2)

        self.get_logger().info(
            f"wrote rollout episode rows={len(self.rows)} csv={self.step_csv} "
            f"summary={self.summary_json}"
        )

    def make_summary(self) -> dict[str, Any]:
        rows = self.rows

        def col(k: str) -> list[float]:
            return [f(r.get(k, 0.0)) for r in rows]

        vx = col("mpc_vx")
        enable = col("mpc_enable")
        h = col("mpc_body_height")
        clr = col("mpc_clearance")
        fallen = col("ram_run_fallen")
        recovery = col("ram_recovery_needed")
        gate_level = col("gate_level_code")
        gate_action = col("gate_action_code")
        override = col("gate_would_override")
        proprio_abs = col("proprio_abs_mean")

        xs = col("odom_x")
        ys = col("odom_y")
        distance_xy = 0.0
        if len(xs) >= 2 and all(math.isfinite(v) for v in [xs[0], ys[0], xs[-1], ys[-1]]):
            dx = xs[-1] - xs[0]
            dy = ys[-1] - ys[0]
            distance_xy = math.sqrt(dx * dx + dy * dy)

        # V0 outcome proxy. This is not a true success label yet.
        # True success/fall should later use Gazebo model state / base height / contact.
        success_proxy = (
            mean(enable) > 0.8
            and mean(vx) > 0.005
            and pct(fallen, 0.90) < 1.05
        )

        return {
            "episode_id": self.episode_id,
            "terrain": self.terrain,
            "policy_id": self.policy_id,
            "duration_sec": self.duration_sec,
            "n_rows": len(rows),

            "mpc_vx_mean": mean(vx),
            "mpc_vx_p50": pct(vx, 0.50),
            "mpc_vx_p90": pct(vx, 0.90),
            "mpc_enable_mean": mean(enable),
            "mpc_body_height_mean": mean(h),
            "mpc_clearance_mean": mean(clr),

            "ram_run_fallen_mean": mean(fallen),
            "ram_run_fallen_p90": pct(fallen, 0.90),
            "ram_recovery_needed_mean": mean(recovery),

            "gate_level_mean": mean(gate_level),
            "gate_action_mean": mean(gate_action),
            "gate_override_mean": mean(override),

            "proprio_abs_mean": mean(proprio_abs),
            "proprio_abs_p90": pct(proprio_abs, 0.90),

            "distance_xy_proxy": distance_xy,
            "success_proxy": bool(success_proxy),

            "step_csv": str(self.step_csv),
            "summary_json": str(self.summary_json),
        }

    def on_timer(self):
        elapsed = time.time() - self.t0
        self.rows.append(self.make_row())

        if elapsed >= self.duration_sec:
            # Data is already flushed to CSV/JSON before exit.
            # Do NOT call rclpy.shutdown() from the timer callback here:
            # on some ROS2/rclpy setups it hangs and the shell script never
            # proceeds to the next rollout episode.
            self.write_outputs()
            print("[TRACER] finished rollout episode recording; hard exit", flush=True)
            os._exit(0)


def main():
    rclpy.init()
    node = RolloutEpisodeRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.write_outputs()
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass


if __name__ == "__main__":
    main()
