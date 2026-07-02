#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def safe_get(xs: Optional[List[float]], i: int, default: float = float("nan")) -> float:
    if xs is None or i >= len(xs):
        return default
    try:
        return float(xs[i])
    except Exception:
        return default


def hypot2(x: float, y: float) -> float:
    if not math.isfinite(x) or not math.isfinite(y):
        return float("nan")
    return math.sqrt(x * x + y * y)


class PhaseBEpisodeRecorder(Node):
    def __init__(self, args):
        super().__init__("tracer_phase_b_episode_recorder_v0")

        self.args = args
        self.t0 = time.monotonic()
        self.wall0 = time.time()

        self.latest_rel: Optional[List[float]] = None
        self.latest_mpc: Optional[List[float]] = None
        self.latest_odom: Optional[List[float]] = None

        self.rows = []

        self.create_subscription(
            Float64MultiArray,
            args.relative_goal_topic,
            self.on_relative_goal,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            args.mpc_reference_topic,
            self.on_mpc_reference,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            args.odom_topic,
            self.on_odom,
            10,
        )

        self.get_logger().info(
            f"recording Phase-B episode for {args.duration:.2f}s -> {args.csv_path}"
        )

    def on_relative_goal(self, msg: Float64MultiArray):
        self.latest_rel = [float(x) for x in msg.data]

    def on_mpc_reference(self, msg: Float64MultiArray):
        self.latest_mpc = [float(x) for x in msg.data]

    def on_odom(self, msg: Float64MultiArray):
        self.latest_odom = [float(x) for x in msg.data]

    def sample_once(self):
        now = time.monotonic()
        t_rel = now - self.t0

        rel_x = safe_get(self.latest_rel, 0)
        rel_y = safe_get(self.latest_rel, 1)
        rel_yaw = safe_get(self.latest_rel, 2)
        rel_enable = safe_get(self.latest_rel, 3, 0.0)
        rel_dist = hypot2(rel_x, rel_y)

        row = {
            "t_rel": t_rel,
            "wall_time": time.time(),
            "rel_x": rel_x,
            "rel_y": rel_y,
            "rel_yaw": rel_yaw,
            "rel_enable": rel_enable,
            "rel_dist": rel_dist,
            "mpc_counter": safe_get(self.latest_mpc, 0),
            "mpc_vx": safe_get(self.latest_mpc, 1),
            "mpc_yaw_rate": safe_get(self.latest_mpc, 2),
            "mpc_body_height": safe_get(self.latest_mpc, 3),
            "mpc_swing_clearance": safe_get(self.latest_mpc, 4),
            "mpc_enable": safe_get(self.latest_mpc, 5),
            "odom_x": safe_get(self.latest_odom, 0),
            "odom_y": safe_get(self.latest_odom, 1),
            "odom_z": safe_get(self.latest_odom, 2),
            "odom_yaw": safe_get(self.latest_odom, 3),
            "odom_vx": safe_get(self.latest_odom, 4),
            "odom_vy": safe_get(self.latest_odom, 5),
        }
        self.rows.append(row)

    def write_outputs(self):
        os.makedirs(os.path.dirname(self.args.csv_path), exist_ok=True)

        fieldnames = [
            "t_rel",
            "wall_time",
            "rel_x",
            "rel_y",
            "rel_yaw",
            "rel_enable",
            "rel_dist",
            "mpc_counter",
            "mpc_vx",
            "mpc_yaw_rate",
            "mpc_body_height",
            "mpc_swing_clearance",
            "mpc_enable",
            "odom_x",
            "odom_y",
            "odom_z",
            "odom_yaw",
            "odom_vx",
            "odom_vy",
        ]

        with open(self.args.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

        valid = [
            r for r in self.rows
            if math.isfinite(r["rel_dist"]) and r["rel_enable"] > 0.5
        ]

        summary = {
            "csv_path": self.args.csv_path,
            "num_rows": len(self.rows),
            "num_valid_goal_rows": len(valid),
            "duration_sec": self.args.duration,
            "stop_distance": self.args.stop_distance,
        }

        if valid:
            first = valid[0]
            last = valid[-1]
            min_row = min(valid, key=lambda r: r["rel_dist"])

            summary.update({
                "initial_rel_dist": first["rel_dist"],
                "final_rel_dist": last["rel_dist"],
                "min_rel_dist": min_row["rel_dist"],
                "progress_initial_minus_final": first["rel_dist"] - last["rel_dist"],
                "progress_initial_minus_min": first["rel_dist"] - min_row["rel_dist"],
                "reached_stop_distance": min_row["rel_dist"] <= self.args.stop_distance,
                "initial_rel_x": first["rel_x"],
                "final_rel_x": last["rel_x"],
                "min_rel_x": min_row["rel_x"],
                "initial_odom_x": first["odom_x"],
                "final_odom_x": last["odom_x"],
                "odom_x_delta": last["odom_x"] - first["odom_x"],
                "max_abs_mpc_yaw_rate": max(abs(r["mpc_yaw_rate"]) for r in valid if math.isfinite(r["mpc_yaw_rate"])),
                "max_mpc_vx": max(r["mpc_vx"] for r in valid if math.isfinite(r["mpc_vx"])),
            })
        else:
            summary.update({
                "reached_stop_distance": False,
                "error": "no valid relative_goal rows",
            })

        with open(self.args.summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--stop-distance", type=float, default=0.15)
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--summary-path", required=True)
    ap.add_argument("--relative-goal-topic", default="/tracer/relative_goal")
    ap.add_argument("--mpc-reference-topic", default="/tracer/mpc_reference")
    ap.add_argument("--odom-topic", default="/tracer/robot_odom_flat")
    args = ap.parse_args()

    rclpy.init()
    node = PhaseBEpisodeRecorder(args)

    dt = 1.0 / max(args.sample_hz, 1e-6)
    end_t = time.monotonic() + args.duration
    next_sample_t = time.monotonic()

    try:
        while rclpy.ok() and time.monotonic() < end_t:
            # Keep callbacks responsive, but sample at the requested fixed rate.
            rclpy.spin_once(node, timeout_sec=min(0.01, dt))

            now = time.monotonic()
            if now >= next_sample_t:
                node.sample_once()
                next_sample_t += dt

                # If callbacks delayed us, avoid a burst of catch-up samples.
                if next_sample_t < now - dt:
                    next_sample_t = now + dt
    finally:
        node.write_outputs()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
