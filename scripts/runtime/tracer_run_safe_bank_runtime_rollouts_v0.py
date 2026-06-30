#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


PROFILE_BETA = {
    "balanced":          [0.34, 0.33, 0.33],
    "motion":            [0.65, 0.20, 0.15],
    "stability":         [0.20, 0.65, 0.15],
    "energy":            [0.20, 0.20, 0.60],
    "motion_extreme":    [0.85, 0.10, 0.05],
    "stability_extreme": [0.05, 0.90, 0.05],
    "energy_extreme":    [0.05, 0.10, 0.85],
}


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd(cmd: list[str], timeout: float = 8.0) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout, p.stderr



def gazebo_stand_reset(model_name: str = "a1_gazebo", z: float = 0.32):
    env = dict(os.environ)
    env["TRACER_GAZEBO_MODEL_NAME"] = str(model_name)
    env["TRACER_GAZEBO_RESET_Z"] = str(z)

    p = subprocess.run(
        ["bash", "scripts/runtime/tracer_gazebo_hard_reset_a1_stand.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15.0,
        env=env,
    )
    return p.returncode, p.stdout, p.stderr


def gazebo_set_model_state(model_name: str = "a1", z: float = 0.42):
    payload = f"""
model_state:
  model_name: '{model_name}'
  pose:
    position: {{x: 0.0, y: 0.0, z: {z}}}
    orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}
  twist:
    linear: {{x: 0.0, y: 0.0, z: 0.0}}
    angular: {{x: 0.0, y: 0.0, z: 0.0}}
  reference_frame: 'world'
"""
    cmd = [
        "docker", "exec", "a1_unitree_gazebo_docker",
        "bash", "--noprofile", "--norc", "-lc",
        (
            "set +u; "
            "source /opt/ros/melodic/setup.bash; "
            "source /root/unitree_ws/devel/setup.bash 2>/dev/null || true; "
            f"timeout 5 rosservice call /gazebo/set_model_state \"{payload}\""
        ),
    ]
    return run_cmd(cmd, timeout=8.0)



def gazebo_call(service: str):
    cmd = [
        "docker", "exec", "a1_unitree_gazebo_docker",
        "bash", "--noprofile", "--norc", "-lc",
        (
            "set +u; "
            "source /opt/ros/melodic/setup.bash; "
            "source /root/unitree_ws/devel/setup.bash 2>/dev/null || true; "
            f'timeout 5 rosservice call {service} "{{}}"'
        ),
    ]
    return run_cmd(cmd, timeout=8.0)


def normalize_beta(beta: list[float]) -> list[float]:
    s = sum(float(x) for x in beta)
    if not math.isfinite(s) or s <= 1e-9:
        return [1.0 / 3.0] * 3
    return [float(x) / s for x in beta]


class SafeBankRuntimeRecorder(Node):
    def __init__(self, profile: str, sample_hz: float):
        super().__init__("tracer_safe_bank_runtime_recorder_v0")

        self.profile = profile
        self.beta = normalize_beta(PROFILE_BETA.get(profile, PROFILE_BETA["balanced"]))

        self.latest_theta: Optional[list[float]] = None
        self.latest_ref: Optional[list[float]] = None
        self.latest_odom: Optional[list[float]] = None
        self.latest_proprio: Optional[list[float]] = None

        self.samples: list[dict] = []

        self.beta_pub = self.create_publisher(
            Float64MultiArray,
            "/tracer/objective_beta",
            10,
        )

        self.phase_pub = self.create_publisher(
            String,
            "/tracer/runtime_phase",
            10,
        )

        self.create_subscription(
            Float64MultiArray,
            "/tracer/selected_theta",
            self._on_theta,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/tracer/mpc_reference",
            self._on_ref,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/tracer/robot_odom_flat",
            self._on_odom,
            10,
        )

        # Optional proprio/state topics. Only one of these may exist depending on
        # the current bridge setup. They are used for base height/attitude if valid.
        for topic in [
            "/tracer/proprio_vector",
            "/tracer/robot_state_flat",
            "/tracer/proprio_flat",
            "/tracer/robot_proprio_flat",
            "/tracer/state_flat",
            "/tracer/base_state_flat",
        ]:
            self.create_subscription(
                Float64MultiArray,
                topic,
                self._on_proprio,
                10,
            )

        self.sample_period = 1.0 / max(1.0, float(sample_hz))
        self._last_sample_t = 0.0
        self.t0 = time.time()

    def publish_beta(self, n: int = 3):
        msg = Float64MultiArray()
        msg.data = self.beta
        for _ in range(n):
            self.beta_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)

    def publish_phase(self, phase: str, n: int = 3):
        msg = String()
        msg.data = str(phase)
        for _ in range(n):
            self.phase_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)

    def _on_theta(self, msg: Float64MultiArray):
        self.latest_theta = [float(x) for x in msg.data]

    def _on_ref(self, msg: Float64MultiArray):
        self.latest_ref = [float(x) for x in msg.data]

    def _on_odom(self, msg: Float64MultiArray):
        self.latest_odom = [float(x) for x in msg.data]

    def _on_proprio(self, msg: Float64MultiArray):
        data = [float(x) for x in msg.data]
        # Accept only non-trivial arrays. A useful robot-state/proprio vector is
        # usually longer than a minimal odom vector.
        if len(data) >= 6:
            self.latest_proprio = data

    @staticmethod
    def _quat_to_rpy(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
        # Quaternion to roll/pitch/yaw.
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    @staticmethod
    def _candidate_base_state(data: list[float]) -> tuple[float, float, float]:
        """Return z, roll, pitch from a flat state-like vector.

        Supported layouts:
        A) /tracer/proprio_vector:
           [stamp, x, y, z, roll, pitch, yaw, ...]

        B) simple pose:
           [x, y, z, roll, pitch, yaw, ...]

        C) odom quaternion:
           [x, y, z, qx, qy, qz, qw, ...]

        Priority matters. proprio_vector has timestamp at index 0 and z at index 3.
        """
        # A) proprio_vector layout: [stamp, x, y, z, roll, pitch, yaw, ...]
        # In current logs, stamp is huge, x/y are world position, z is index 3.
        if len(data) >= 7:
            stamp = float(data[0])
            z = float(data[3])
            roll = float(data[4])
            pitch = float(data[5])
            if stamp > 1.0e6 and 0.12 <= z <= 0.70 and abs(roll) <= 1.5 and abs(pitch) <= 1.5:
                return z, roll, pitch

        # C) odom quaternion layout: [x, y, z, qx, qy, qz, qw, ...]
        if len(data) >= 7:
            z = float(data[2])
            qx, qy, qz, qw = map(float, data[3:7])
            qnorm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)

            if 0.12 <= z <= 0.70 and 0.70 <= qnorm <= 1.30:
                qx, qy, qz, qw = qx / qnorm, qy / qnorm, qz / qnorm, qw / qnorm
                roll, pitch, _ = SafeBankRuntimeRecorder._quat_to_rpy(qx, qy, qz, qw)
                return z, roll, pitch

        # B) simple pose layout: [x, y, z, roll, pitch, yaw, ...]
        if len(data) >= 6:
            z = float(data[2])
            roll = float(data[3])
            pitch = float(data[4])
            if 0.12 <= z <= 0.70 and abs(roll) <= 1.5 and abs(pitch) <= 1.5:
                return z, roll, pitch

        # Fallback: find plausible base height, but do not trust attitude.
        for val in data[:12]:
            v = float(val)
            if 0.18 <= v <= 0.45:
                return v, 0.0, 0.0

        return float("nan"), float("nan"), float("nan")

    def maybe_sample(self):
        t = time.time()
        if t - self._last_sample_t < self.sample_period:
            return
        self._last_sample_t = t

        odom = self.latest_odom or []
        proprio = self.latest_proprio or []
        theta = self.latest_theta or []
        ref = self.latest_ref or []

        # Odom is used mainly for x/y travel distance.
        x = odom[0] if len(odom) > 0 else float("nan")
        y = odom[1] if len(odom) > 1 else float("nan")
        yaw = odom[5] if len(odom) > 5 else float("nan")

        # Base height/attitude should come from proprio/state if available.
        z, roll, pitch = self._candidate_base_state(proprio)
        state_source = "proprio"

        if not math.isfinite(z):
            z, roll, pitch = self._candidate_base_state(odom)
            state_source = "odom"

        self.samples.append({
            "t": t - self.t0,
            "profile": self.profile,
            "beta": json.dumps(self.beta),
            "theta_norm": json.dumps(theta),
            "mpc_reference": json.dumps(ref),
            "odom": json.dumps(odom),
            "proprio": json.dumps(proprio),
            "state_source": state_source,
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        })


def finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def _is_policy_sample(sample: dict) -> bool:
    """Return True if the sample is from the active policy phase.

    Hold/prestand refs use vx=0.0, body_height=0.300, clearance=0.035.
    Policy refs should have nonzero vx for the current safe-bank anchors.
    """
    try:
        ref = json.loads(sample.get("mpc_reference", "[]"))
        if len(ref) < 6:
            return False
        return abs(float(ref[1])) > 1.0e-5
    except Exception:
        return False


def summarize_episode(profile: str, samples: list[dict], metric_warmup_sec: float = 0.0) -> dict:
    # Exclude reset/unpause transient and hold/prestand samples from metrics.
    eval_samples = [
        s for s in samples
        if float(s.get("t", 0.0)) >= float(metric_warmup_sec)
        and _is_policy_sample(s)
    ]

    valid = len(eval_samples) > 0 and any(finite(float(s["z"])) for s in eval_samples)

    xs = [float(s["x"]) for s in eval_samples if finite(float(s["x"]))]
    ys = [float(s["y"]) for s in eval_samples if finite(float(s["y"]))]
    zs = [float(s["z"]) for s in eval_samples if finite(float(s["z"]))]
    rolls = [abs(float(s["roll"])) for s in eval_samples if finite(float(s["roll"]))]
    pitches = [abs(float(s["pitch"])) for s in eval_samples if finite(float(s["pitch"]))]

    if len(xs) >= 2 and len(ys) >= 2:
        distance = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    else:
        distance = 0.0

    z_min = min(zs) if zs else 0.0
    z_mean = sum(zs) / len(zs) if zs else 0.0
    roll_max = max(rolls) if rolls else 999.0
    pitch_max = max(pitches) if pitches else 999.0

    # Keep same conservative stability flag used by scoring layer.
    height_stable = bool(valid and z_min >= 0.18)

    latest_theta = []
    latest_ref = []
    for s in reversed(samples):
        try:
            latest_theta = json.loads(s["theta_norm"])
            latest_ref = json.loads(s["mpc_reference"])
            if latest_ref:
                break
        except Exception:
            pass

    # Current ROS1 bridge mpc ref layout:
    # [counter, vx, yaw_rate, body_height, swing_clearance, enable]
    vx_cmd = float(latest_ref[1]) if len(latest_ref) > 1 else 0.0
    yaw_rate_cmd = float(latest_ref[2]) if len(latest_ref) > 2 else 0.0
    body_height_cmd = float(latest_ref[3]) if len(latest_ref) > 3 else 0.30
    clearance_cmd = float(latest_ref[4]) if len(latest_ref) > 4 else 0.035

    # Command-based effort proxy until torque/current logging is wired.
    # Lower is better.
    command_effort_proxy = (
        0.45 * abs(vx_cmd)
        + 0.10 * abs(yaw_rate_cmd)
        + 0.20 * abs(body_height_cmd - 0.30)
        + 0.25 * abs(clearance_cmd)
    )

    return {
        "preset": f"safe_bank_runtime_{profile}",
        "terrain": "flat_normal",
        "profile": profile,
        "theta_norm": json.dumps(latest_theta),
        "mpc_reference": json.dumps(latest_ref),
        "valid_data": str(bool(valid)),
        "proprio_base_height_stable": str(bool(height_stable)),
        "distance_xy_proxy": distance,
        "proprio_base_z_min": z_min,
        "proprio_base_z_mean": z_mean,
        "proprio_roll_abs_max": roll_max,
        "proprio_pitch_abs_max": pitch_max,
        "vx_cmd": vx_cmd,
        "yaw_rate_cmd": yaw_rate_cmd,
        "body_height_cmd": body_height_cmd,
        "clearance_cmd": clearance_cmd,
        "command_effort_proxy": command_effort_proxy,
        "num_samples": len(samples),
        "num_eval_samples": len(eval_samples),
        "metric_warmup_sec": float(metric_warmup_sec),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="balanced", choices=list(PROFILE_BETA))
    ap.add_argument("--duration-sec", type=float, default=3.0)
    ap.add_argument("--sample-hz", type=float, default=10.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out-root", default="data/rollouts/safe_bank_runtime_v0")
    ap.add_argument("--control-gazebo", action="store_true")
    ap.add_argument("--reset-gazebo", action="store_true")
    ap.add_argument("--hard-reset-gazebo", action="store_true")
    ap.add_argument("--stand-reset-gazebo", action="store_true")
    ap.add_argument("--reset-model-name", default="a1_gazebo")
    ap.add_argument("--reset-z", type=float, default=0.33)
    ap.add_argument("--prestand-sec", type=float, default=1.5)
    ap.add_argument("--reset-settle-sec", type=float, default=0.5)
    ap.add_argument("--metric-warmup-sec", type=float, default=0.8)
    args = ap.parse_args()

    out_dir = Path(args.out_root) / f"safe_bank_runtime_v0_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = SafeBankRuntimeRecorder(args.profile, args.sample_hz)

    summaries = []

    try:
        for ep in range(args.repeats):
            print("=" * 80)
            print(f"[TRACER] episode {ep+1}/{args.repeats} profile={args.profile}")

            node.samples = []
            node.t0 = time.time()
            node.publish_beta(n=5)

            if args.control_gazebo and (args.reset_gazebo or args.hard_reset_gazebo or args.stand_reset_gazebo):
                print("[TRACER] pause gazebo before reset")
                code, out, err = gazebo_call("/gazebo/pause_physics")
                if out.strip():
                    print(out.strip())
                if err.strip():
                    print(err.strip())

                if args.reset_gazebo:
                    print("[TRACER] reset gazebo world")
                    code, out, err = gazebo_call("/gazebo/reset_world")
                    if out.strip():
                        print(out.strip())
                    if err.strip():
                        print(err.strip())

                if args.hard_reset_gazebo:
                    print("[TRACER] hard reset model pose")
                    code, out, err = gazebo_set_model_state(
                        model_name=str(args.reset_model_name),
                        z=float(args.reset_z),
                    )
                    if out.strip():
                        print(out.strip())
                    if err.strip():
                        print(err.strip())

                if args.stand_reset_gazebo:
                    print("[TRACER] hard reset model + joints to stand")
                    code, out, err = gazebo_stand_reset(
                        model_name=str(args.reset_model_name),
                        z=float(args.reset_z),
                    )
                    if out.strip():
                        print(out.strip())
                    if err.strip():
                        print(err.strip())

                time.sleep(max(0.0, float(args.reset_settle_sec)))

                # Hold/stand phase before policy rollout.
                print("[TRACER] runtime phase: hold")
                node.publish_phase("hold", n=5)
                node.publish_beta(n=5)

                if args.control_gazebo:
                    print("[TRACER] unpause gazebo for pre-stand")
                    code, out, err = gazebo_call("/gazebo/unpause_physics")
                    if out.strip():
                        print(out.strip())
                    if err.strip():
                        print(err.strip())

                    t_pre = time.time() + max(0.0, float(args.prestand_sec))
                    while time.time() < t_pre:
                        rclpy.spin_once(node, timeout_sec=0.02)
                        node.maybe_sample()

                    print("[TRACER] pause gazebo after pre-stand")
                    code, out, err = gazebo_call("/gazebo/pause_physics")
                    if out.strip():
                        print(out.strip())
                    if err.strip():
                        print(err.strip())

                print("[TRACER] runtime phase: policy")
                node.publish_phase("policy", n=10)

                # Wait until /tracer/mpc_reference has actually switched from hold to policy.
                print("[TRACER] waiting for policy ref")
                deadline = time.time() + 1.0
                while time.time() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.02)
                    ref = node.latest_ref or []
                    if len(ref) >= 6 and abs(float(ref[1])) > 1.0e-5:
                        break

                # Start actual evaluation after pre-stand and policy-ref switch.
                node.t0 = time.time()
                node.samples = []

            if args.control_gazebo:
                print("[TRACER] unpause gazebo")
                code, out, err = gazebo_call("/gazebo/unpause_physics")
                print(out.strip())
                if err.strip():
                    print(err.strip())

            t_end = time.time() + float(args.duration_sec)
            while time.time() < t_end:
                rclpy.spin_once(node, timeout_sec=0.02)
                node.maybe_sample()

            if args.control_gazebo:
                print("[TRACER] pause gazebo")
                code, out, err = gazebo_call("/gazebo/pause_physics")
                print(out.strip())
                if err.strip():
                    print(err.strip())

            raw_path = out_dir / f"episode_{ep:03d}_samples.csv"
            if node.samples:
                with raw_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(node.samples[0].keys()))
                    writer.writeheader()
                    writer.writerows(node.samples)
            else:
                raw_path.write_text("")

            summary = summarize_episode(
                args.profile,
                node.samples,
                metric_warmup_sec=float(args.metric_warmup_sec),
            )
            summary["episode"] = ep
            summaries.append(summary)

            print(
                "[TRACER] summary "
                f"valid={summary['valid_data']} "
                f"stable={summary['proprio_base_height_stable']} "
                f"dist={float(summary['distance_xy_proxy']):.4f} "
                f"zmin={float(summary['proprio_base_z_min']):.4f} "
                f"roll={float(summary['proprio_roll_abs_max']):.4f} "
                f"pitch={float(summary['proprio_pitch_abs_max']):.4f} "
                f"samples={summary['num_samples']}"
            )

        summary_path = out_dir / "theta_sweep_summary.csv"
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

        meta = {
            "profile": args.profile,
            "duration_sec": args.duration_sec,
            "sample_hz": args.sample_hz,
            "repeats": args.repeats,
            "control_gazebo": args.control_gazebo,
            "out_dir": str(out_dir),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        print()
        print("[TRACER] wrote:", out_dir)
        print("[TRACER] summary_csv:", summary_path)

    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
