#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_DATASET = "data/rollout_metrics/meta_gait_v0_gms_dataset.csv"


def as_float(x, default=0.0):
    try:
        if x == "":
            return default
        return float(x)
    except Exception:
        return default


def as_bool(x):
    return str(x).strip().lower() in ("1", "true", "yes", "y")


def load_rows(path: str):
    return list(csv.DictReader(Path(path).open()))


def find_row(rows, name: str):
    for row in rows:
        if row.get("name") == name:
            return row
    raise SystemExit(f"No terrain row found for name={name!r}")


def build_lowlevel_ref(row: dict, args) -> dict:
    vx_scale = as_float(row.get("vx_scale"), 1.0)
    body_height_delta = as_float(row.get("body_height_delta"), 0.0)
    clearance_delta = as_float(row.get("clearance_delta"), 0.0)
    impedance_scale = as_float(row.get("impedance_scale"), 1.0)
    recovery_needed = as_bool(row.get("recovery_needed"))

    nominal_vx = float(args.nominal_vx)
    nominal_body_height = float(args.nominal_body_height)
    nominal_clearance = float(args.nominal_clearance)

    vx = nominal_vx * vx_scale
    body_height = nominal_body_height + body_height_delta
    clearance = nominal_clearance + clearance_delta

    # If GMS marks the terrain as no-valid-forward / recovery-needed critical,
    # keep forward command at zero. This is a scaffold for later recovery primitives.
    if recovery_needed and row.get("semantic_mode") == "no_valid_forward_recovery_needed":
        vx = 0.0

    # Current TRACER MPC reference convention used in previous bridge:
    # [yaw_rate, vx, vy, body_height, swing_clearance, enable]
    mpc_array = [
        0.0,
        vx,
        0.0,
        body_height,
        clearance,
        1.0,
    ]

    return {
        "terrain_name": row.get("name"),
        "label": row.get("label"),
        "semantic_mode": row.get("semantic_mode"),
        "risk_level": row.get("risk_level"),
        "recovery_needed": recovery_needed,
        "meta_gait_proposal": {
            "vx": vx,
            "body_height": body_height,
            "swing_clearance": clearance,
            "impedance_scale": impedance_scale,
            "vx_scale": vx_scale,
            "body_height_delta": body_height_delta,
            "clearance_delta": clearance_delta,
        },
        "lowlevel_reference": {
            "mpc_array": mpc_array,
            "mpc_array_layout": [
                "yaw_rate",
                "vx",
                "vy",
                "body_height",
                "swing_clearance",
                "enable",
            ],
        },
        "source_metrics": {
            "reward_mean": as_float(row.get("reward_mean")),
            "directional_dx": as_float(row.get("directional_dx")),
            "abs_final_dy": as_float(row.get("abs_final_dy")),
            "done_count": int(as_float(row.get("done_count"))),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--name", required=True)
    ap.add_argument("--nominal_vx", type=float, default=0.10)
    ap.add_argument("--nominal_body_height", type=float, default=0.295)
    ap.add_argument("--nominal_clearance", type=float, default=0.030)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.dataset)
    row = find_row(rows, args.name)
    ref = build_lowlevel_ref(row, args)

    if args.json:
        print(json.dumps(ref, indent=2, sort_keys=True))
        return

    mg = ref["meta_gait_proposal"]
    ll = ref["lowlevel_reference"]

    print(f"terrain_name: {ref['terrain_name']}")
    print(f"label: {ref['label']}")
    print(f"semantic_mode: {ref['semantic_mode']}")
    print(f"risk_level: {ref['risk_level']}")
    print(f"recovery_needed: {ref['recovery_needed']}")
    print(
        "meta_gait_proposal: "
        f"vx={mg['vx']:.4f}, "
        f"body_height={mg['body_height']:.4f}, "
        f"swing_clearance={mg['swing_clearance']:.4f}, "
        f"impedance_scale={mg['impedance_scale']:.4f}"
    )
    print(f"mpc_array: {ll['mpc_array']}")


if __name__ == "__main__":
    main()
