#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-name", required=True)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--block-exit-code", type=int, default=20)
    args = ap.parse_args()

    raw = subprocess.check_output([
        "python3",
        "scripts/runtime/tracer_route_phase_b_theta_extended_v0.py",
        "--world-name",
        args.world_name,
    ], text=True)

    route = json.loads(raw)
    target = route.get("theta_extended_target") or {}

    exec_now = bool(target.get("executable_by_current_a1_qpmc", False))
    requires_phase = bool(target.get("requires_lowlevel_phase_interface", True))
    theta = target.get("theta_extended") or {}

    out = {
        "schema": "phase_b_theta_extended_executability_check_v0",
        "world_name": args.world_name,
        "target_primitive_family": route.get("target_primitive_family"),
        "theta_extended_available": bool(route.get("theta_extended_available", False)),
        "executable_by_current_a1_qpmc": exec_now,
        "requires_lowlevel_phase_interface": requires_phase,
        "blocked_by_current_lowlevel": not exec_now,
        "gait_family": theta.get("gait_family"),
        "theta_extended": theta,
        "reason": (
            "current A1-QP-MPC theta-lite interface can execute this primitive"
            if exec_now
            else "primitive requires low-level gait/phase interface not available in current A1-QP-MPC bridge"
        ),
        "route": route,
    }

    text = json.dumps(out, indent=2, sort_keys=True)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n")

    print(text)

    if not exec_now:
        sys.exit(args.block_exit_code)


if __name__ == "__main__":
    main()
