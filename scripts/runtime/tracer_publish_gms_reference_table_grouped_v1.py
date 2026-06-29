#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def get_ref(table: str, terrain: str, mode: str, bucket: str) -> list[float]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/runtime/tracer_query_gms_reference_table_grouped_v1.py"),
        "--table", table,
        "--terrain", terrain,
        "--mode", mode,
        "--bucket", bucket,
        "--json",
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    return [float(x) for x in data["mpc_reference"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/rollout_metrics/gms_reference_table_grouped_v1.json")
    ap.add_argument("--terrain", default="flat_normal")
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--bucket", default="preferred")
    ap.add_argument("--hz", type=float, default=20.0)
    args = ap.parse_args()

    ref = get_ref(args.table, args.terrain, args.mode, args.bucket)
    msg = "{data: [" + ", ".join(f"{x:.6g}" for x in ref) + "]}"

    print("[TRACER] selected ref:", ref)
    print("[TRACER] publishing /tracer/mpc_reference at", args.hz, "Hz")
    print("[TRACER] msg:", msg)

    cmd = [
        "bash", "-lc",
        f'''
set +u
source /opt/ros/humble/setup.bash
if [ -f ros2_ws/install/setup.bash ]; then source ros2_ws/install/setup.bash; fi
ros2 topic pub /tracer/mpc_reference std_msgs/msg/Float64MultiArray "{msg}" -r {args.hz}
'''
    ]

    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
