#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[4]


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE_FIELDS = [
    "stamp_wall",
    "ok",
    "risk",
    "bad_prob",
    "good_prob",
    "age_sec",
    "error",
]

ROW_FIELDS = [
    "mpc_vx",
    "mpc_yaw",
    "mpc_body_height",
    "mpc_clearance",
    "mpc_enable",
    "beta_motion",
    "beta_stability",
    "beta_energy",
    "proprio_abs_mean",
    "proprio_abs_max",
    "proprio_base_x",
    "proprio_base_y",
    "proprio_base_z",
    "odom_x",
    "odom_y",
    "odom_z",
    "odom_vx",
    "age_mpc",
    "age_beta",
    "age_proprio",
    "age_odom",
    "age_debug",
]


class RAMScalarV3ShadowCsvLogger(Node):
    def __init__(self):
        super().__init__("tracer_ram_scalar_v3_shadow_csv_logger")

        default_dir = ROOT / "data" / "ram_scalar_v3_shadow_logs"
        self.declare_parameter("topic", os.environ.get("TRACER_RAM_SCALAR_V3_SHADOW_TOPIC", "/tracer/ram_scalar_v3_shadow"))
        self.declare_parameter("out_dir", os.environ.get("TRACER_RAM_SCALAR_V3_SHADOW_LOG_DIR", str(default_dir)))
        self.declare_parameter("run_name", os.environ.get("TRACER_RAM_SCALAR_V3_SHADOW_RUN_NAME", ""))
        self.declare_parameter("max_rows", int(os.environ.get("TRACER_RAM_SCALAR_V3_SHADOW_MAX_ROWS", "0")))

        self.topic = str(self.get_parameter("topic").value)
        self.max_rows = int(self.get_parameter("max_rows").value)
        out_dir = Path(str(self.get_parameter("out_dir").value))
        out_dir.mkdir(parents=True, exist_ok=True)

        run_name = str(self.get_parameter("run_name").value).strip()
        if not run_name:
            run_name = time.strftime("ram_scalar_v3_shadow_%Y%m%d_%H%M%S")

        self.out_path = out_dir / f"{run_name}.csv"
        self.fp = self.out_path.open("w", newline="")
        self.fields = BASE_FIELDS + [f"row_{x}" for x in ROW_FIELDS]
        self.writer = csv.DictWriter(self.fp, fieldnames=self.fields)
        self.writer.writeheader()
        self.count = 0
        self.stop_requested = False

        self.create_subscription(String, self.topic, self.on_msg, 50)

        self.get_logger().info(
            f"RAM scalar v3 shadow CSV logger started topic={self.topic} out={self.out_path} "
            f"max_rows={self.max_rows}"
        )

    def on_msg(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"failed to parse shadow json: {exc}")
            return

        row_in = payload.get("row", {}) or {}
        out = {}
        for k in BASE_FIELDS:
            out[k] = payload.get(k, "")
        for k in ROW_FIELDS:
            out[f"row_{k}"] = row_in.get(k, "")

        self.writer.writerow(out)
        self.count += 1

        if self.count % 100 == 0:
            self.fp.flush()
            self.get_logger().info(f"logged {self.count} RAM scalar v3 shadow rows to {self.out_path}")

        if self.max_rows > 0 and self.count >= self.max_rows:
            self.fp.flush()
            self.get_logger().info(
                f"max_rows reached: {self.count}; stopping RAM scalar v3 shadow CSV logger"
            )
            self.stop_requested = True

    def destroy_node(self):
        try:
            self.fp.flush()
            self.fp.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = RAMScalarV3ShadowCsvLogger()
    try:
        while rclpy.ok() and not node.stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
