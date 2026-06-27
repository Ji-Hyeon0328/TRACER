#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def f(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def terrain_onehot(name: str):
    return [
        1.0 if name == "flat_normal" else 0.0,
        1.0 if name == "rough_mid" else 0.0,
        1.0 if name == "slope_5deg" else 0.0,
    ]


def beta_target_from_terrain(name: str):
    if name == "flat_normal":
        return [0.55, 0.25, 0.20]
    if name == "rough_mid":
        return [0.30, 0.50, 0.20]
    if name == "slope_5deg":
        return [0.25, 0.55, 0.20]
    return [0.25, 0.55, 0.20]


def build_x(row):
    terrain = row.get("terrain", "unknown")

    return terrain_onehot(terrain) + [
        f(row.get("duration_sec", 0.0)) / 30.0,
        f(row.get("mpc_vx_mean", 0.0)),
        f(row.get("mpc_enable_mean", 0.0)),
        f(row.get("mpc_body_height_mean", 0.0)),
        f(row.get("mpc_clearance_mean", 0.0)),
        f(row.get("ram_run_fallen_mean", 0.0)),
        f(row.get("ram_recovery_needed_mean", 0.0)),
        f(row.get("gate_level_mean", 0.0)),
        f(row.get("gate_action_mean", 0.0)),
        f(row.get("gate_override_mean", 0.0)),
        f(row.get("distance_xy_proxy", 0.0)) / 40.0,
        1.0 if str(row.get("success_proxy", "")).lower() == "true" else 0.0,
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", default="data/rollout_dataset_v0/rollout_summary_table_v0.csv")
    ap.add_argument("--out", default="data/objective_selector_dataset_v1/objective_selector_dataset_v1.jsonl")
    ap.add_argument("--min-duration", type=float, default=20.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.summary_csv) as fobj:
        for r in csv.DictReader(fobj):
            if f(r.get("duration_sec", 0.0)) < args.min_duration:
                continue
            terrain = r.get("terrain", "unknown")
            item = {
                "episode_id": r.get("episode_id", "unknown"),
                "terrain": terrain,
                "x": build_x(r),
                "beta": beta_target_from_terrain(terrain),
                "beta_names": ["motion", "stability", "energy"],
            }
            rows.append(item)

    by_terrain = {}
    for r in rows:
        by_terrain[r["terrain"]] = by_terrain.get(r["terrain"], 0) + 1

    with out.open("w") as fobj:
        for r in rows:
            fobj.write(json.dumps(r) + "\n")

    manifest = {
        "out": str(out),
        "summary_csv": args.summary_csv,
        "n_samples": len(rows),
        "input_dim": len(rows[0]["x"]) if rows else 0,
        "output_dim": 3,
        "beta_names": ["motion", "stability", "energy"],
        "by_terrain": by_terrain,
    }

    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[TRACER] wrote objective selector dataset: {out}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
