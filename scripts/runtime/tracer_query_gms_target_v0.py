#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_DATASET = "data/rollout_metrics/meta_gait_v0_gms_dataset.csv"


def load_rows(path: str):
    return list(csv.DictReader(Path(path).open()))


def as_float(x, default=0.0):
    try:
        if x == "":
            return default
        return float(x)
    except Exception:
        return default


def as_bool(x):
    return str(x).strip().lower() in ("1", "true", "yes", "y")


def find_by_name(rows, name: str):
    for row in rows:
        if row.get("name") == name:
            return row
    return None


def find_by_descriptor(rows, args):
    candidates = []
    for row in rows:
        if args.material and row.get("material") != args.material:
            continue
        if args.slope_type and row.get("slope_type") != args.slope_type:
            continue
        if args.geometry and row.get("geometry") != args.geometry:
            continue
        if args.gait_direction and row.get("gait_direction") != args.gait_direction:
            continue
        if args.slope_deg is not None:
            if abs(as_float(row.get("slope_deg")) - float(args.slope_deg)) > 1.0e-6:
                continue
        if args.rough_start_x is not None:
            if row.get("rough_start_x", "") == "":
                continue
            if abs(as_float(row.get("rough_start_x")) - float(args.rough_start_x)) > 1.0e-6:
                continue
        candidates.append(row)

    if not candidates:
        return None
    if len(candidates) > 1:
        names = ", ".join(r["name"] for r in candidates)
        raise RuntimeError(f"Descriptor matched multiple rows: {names}")
    return candidates[0]


def compact_target(row):
    return {
        "name": row.get("name"),
        "material": row.get("material"),
        "slope_type": row.get("slope_type"),
        "slope_deg": as_float(row.get("slope_deg")),
        "geometry": row.get("geometry"),
        "rough_start_x": row.get("rough_start_x"),
        "gait_direction": row.get("gait_direction"),
        "label": row.get("label"),
        "semantic_mode": row.get("semantic_mode"),
        "risk_level": row.get("risk_level"),
        "vx_scale": as_float(row.get("vx_scale")),
        "body_height_delta": as_float(row.get("body_height_delta")),
        "clearance_delta": as_float(row.get("clearance_delta")),
        "impedance_scale": as_float(row.get("impedance_scale")),
        "recovery_needed": as_bool(row.get("recovery_needed")),
        "reward_mean": as_float(row.get("reward_mean")),
        "directional_dx": as_float(row.get("directional_dx")),
        "abs_final_dy": as_float(row.get("abs_final_dy")),
        "done_count": int(as_float(row.get("done_count"))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--name", default="")
    ap.add_argument("--material", default="")
    ap.add_argument("--slope_type", default="")
    ap.add_argument("--slope_deg", type=float, default=None)
    ap.add_argument("--geometry", default="")
    ap.add_argument("--rough_start_x", type=float, default=None)
    ap.add_argument("--gait_direction", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.dataset)

    if args.name:
        row = find_by_name(rows, args.name)
    else:
        row = find_by_descriptor(rows, args)

    if row is None:
        raise SystemExit("No matching GMS target found.")

    target = compact_target(row)

    if args.json:
        print(json.dumps(target, indent=2, sort_keys=True))
        return

    print(f"name: {target['name']}")
    print(f"terrain: material={target['material']} slope={target['slope_type']}({target['slope_deg']}) geometry={target['geometry']} gait={target['gait_direction']}")
    print(f"label: {target['label']}")
    print(f"semantic_mode: {target['semantic_mode']}")
    print(f"risk_level: {target['risk_level']}")
    print(f"vx_scale: {target['vx_scale']}")
    print(f"body_height_delta: {target['body_height_delta']}")
    print(f"clearance_delta: {target['clearance_delta']}")
    print(f"impedance_scale: {target['impedance_scale']}")
    print(f"recovery_needed: {target['recovery_needed']}")
    print(f"metrics: reward_mean={target['reward_mean']} directional_dx={target['directional_dx']} abs_final_dy={target['abs_final_dy']} done_count={target['done_count']}")


if __name__ == "__main__":
    main()
