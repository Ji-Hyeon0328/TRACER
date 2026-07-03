#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-name", required=True)
    ap.add_argument("--seed-json", default="configs/phase_b_primitive_theta_extended_v0/primitive_theta_extended_seed_v0.json")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    route_raw = subprocess.check_output([
        "python3",
        "scripts/runtime/tracer_route_phase_b_primitive_v0.py",
        "--world-name",
        args.world_name,
    ], text=True)
    route = json.loads(route_raw)

    seeds = json.load(open(args.seed_json))
    families = seeds["primitive_families"]

    target_family = route.get("target_primitive_family")
    seed = families.get(target_family)

    out = {
        "schema": "phase_b_theta_extended_route_v0",
        "world_name": args.world_name,
        "primitive_route": route,
        "target_primitive_family": target_family,
        "theta_extended_available": seed is not None,
        "theta_extended_target": None,
    }

    if seed is not None:
        out["theta_extended_target"] = {
            "primitive_family": target_family,
            "description": seed.get("description"),
            "executable_by_current_a1_qpmc": seed.get("executable_by_current_a1_qpmc", False),
            "requires_lowlevel_phase_interface": seed.get("requires_lowlevel_phase_interface", True),
            "theta_extended": seed["theta_extended"],
        }

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
