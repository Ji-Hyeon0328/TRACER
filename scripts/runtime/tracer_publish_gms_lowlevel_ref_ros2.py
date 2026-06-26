#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


DEFAULT_TABLE = "data/rollout_metrics/meta_gait_v0_lowlevel_refs.csv"


def load_rows(path: str):
    return list(csv.DictReader(Path(path).open()))


def find_row(rows, name: str):
    for row in rows:
        if row.get("name") == name:
            return row
    available = ", ".join(r.get("name", "") for r in rows)
    raise SystemExit(f"No row found for terrain_name={name!r}. Available: {available}")


def parse_mpc_array(row: dict):
    raw = row.get("mpc_array_json", "").strip()
    if raw:
        return [float(x) for x in json.loads(raw)]

    return [
        float(row["mpc_yaw_rate"]),
        float(row["mpc_vx"]),
        float(row["mpc_vy"]),
        float(row["mpc_body_height"]),
        float(row["mpc_swing_clearance"]),
        float(row["mpc_enable"]),
    ]


def format_summary(row: dict, mpc_array):
    return (
        f"terrain={row['name']} "
        f"label={row['label']} "
        f"mode={row['semantic_mode']} "
        f"risk={row['risk_level']} "
        f"recovery={row['recovery_needed']} "
        f"mpc={mpc_array}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--terrain_name", default=os.environ.get("TRACER_TERRAIN_NAME", "solid_even_forward"))
    ap.add_argument("--topic", default="/tracer/mpc_reference")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.table)
    row = find_row(rows, args.terrain_name)
    mpc_array = parse_mpc_array(row)

    print(format_summary(row, mpc_array), flush=True)

    if args.dry_run:
        return

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray

    class GmsLowlevelRefPublisher(Node):
        def __init__(self):
            super().__init__("tracer_gms_lowlevel_ref_publisher_v0")
            self.pub = self.create_publisher(Float64MultiArray, args.topic, 10)
            self.msg = Float64MultiArray()
            self.msg.data = mpc_array
            period = 1.0 / max(args.hz, 1.0e-6)
            self.timer = self.create_timer(period, self._on_timer)
            self.count = 0
            self.get_logger().info(format_summary(row, mpc_array))
            self.get_logger().info(f"publishing to {args.topic} at {args.hz} Hz")

        def _on_timer(self):
            self.pub.publish(self.msg)
            self.count += 1
            if args.once and self.count >= 1:
                rclpy.shutdown()

    rclpy.init()
    node = GmsLowlevelRefPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
