#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def label_group(r) -> str:
    if r["n"] <= 0 or r["valid_rate"] < 0.8:
        return "invalid_or_insufficient"

    if r["bad_rate"] >= 0.34:
        return "avoid_variable_or_bad"

    if r["mean_cost"] <= -0.15 and r["mean_dx"] >= 0.12 and r["mean_z_p10"] >= 0.32:
        return "preferred_fast"

    if r["mean_cost"] <= 0.0 and r["mean_dx"] >= 0.08 and r["mean_z_p10"] >= 0.32:
        return "preferred_balanced"

    if r["mean_cost"] <= 0.5 and r["mean_dx"] >= 0.06 and r["mean_z_p10"] >= 0.30:
        return "acceptable"

    return "avoid_or_unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="data/rollout_metrics/gms_suitability_dataset_v1.csv")
    ap.add_argument("--out-json", default="data/rollout_metrics/gms_reference_table_grouped_v1.json")
    ap.add_argument("--out-csv", default="data/rollout_metrics/gms_reference_table_grouped_v1.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # Runtime table should use only attitude-aware samples.
    df = df[df["has_attitude"] == True].copy()

    bad_labels = {
        "invalid_data",
        "negative_lateral_drift",
        "negative_low_height",
        "negative_unstable_attitude",
        "variable_forward_with_lateral_drift",
    }

    rows = []
    group_cols = ["terrain", "vx", "body_height", "clearance"]

    for keys, g in df.groupby(group_cols):
        terrain, vx, h, clr = keys

        n = len(g)
        valid_rate = float(g["valid_data"].mean())
        bad_rate = float(g["suitability_label"].isin(bad_labels).mean())

        r = {
            "terrain": terrain,
            "vx": float(vx),
            "body_height": float(h),
            "clearance": float(clr),
            "n": int(n),
            "valid_rate": valid_rate,
            "bad_rate": bad_rate,
            "mean_cost": float(g["cost"].mean()),
            "std_cost": float(g["cost"].std()) if n > 1 else 0.0,
            "mean_dx": float(g["dx"].mean()),
            "std_dx": float(g["dx"].std()) if n > 1 else 0.0,
            "mean_z_p10": float(g["z_p10"].mean()),
            "mean_abs_dy_per_dx": float(g["abs_dy_per_dx"].mean()),
            "mean_roll_abs_max": float(g["roll_abs_max"].mean()),
            "mean_pitch_abs_max": float(g["pitch_abs_max"].mean()),
            "mean_abs_yaw_delta": float(g["yaw_delta"].abs().mean()),
            "labels": dict(g["suitability_label"].value_counts()),
            "episodes": list(g["episode_id"].astype(str)),
        }

        r["group_label"] = label_group(r)
        rows.append(r)

    out = pd.DataFrame(rows).sort_values(["terrain", "mean_cost"])

    table = {
        "version": "gms_reference_table_grouped_v1",
        "source": args.in_csv,
        "terrains": {},
    }

    for terrain, g in out.groupby("terrain"):
        gg = g.sort_values("mean_cost")

        preferred = gg[gg["group_label"].isin(["preferred_fast", "preferred_balanced"])].copy()
        acceptable = gg[gg["group_label"].isin(["acceptable"])].copy()
        avoid = gg[gg["group_label"].str.startswith("avoid") | gg["group_label"].str.startswith("invalid")].copy()

        def items(frame):
            result = []
            for _, r in frame.iterrows():
                result.append({
                    "vx": float(r["vx"]),
                    "body_height": float(r["body_height"]),
                    "clearance": float(r["clearance"]),
                    "group_label": str(r["group_label"]),
                    "n": int(r["n"]),
                    "valid_rate": float(r["valid_rate"]),
                    "bad_rate": float(r["bad_rate"]),
                    "mean_cost": float(r["mean_cost"]),
                    "std_cost": float(r["std_cost"]),
                    "mean_dx": float(r["mean_dx"]),
                    "std_dx": float(r["std_dx"]),
                    "mean_z_p10": float(r["mean_z_p10"]),
                    "mean_abs_dy_per_dx": float(r["mean_abs_dy_per_dx"]),
                    "mean_roll_abs_max": float(r["mean_roll_abs_max"]),
                    "mean_pitch_abs_max": float(r["mean_pitch_abs_max"]),
                    "mean_abs_yaw_delta": float(r["mean_abs_yaw_delta"]),
                })
            return result

        table["terrains"][terrain] = {
            "preferred": items(preferred),
            "acceptable": items(acceptable),
            "avoid": items(avoid),
        }

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(table, indent=2), encoding="utf-8")

    print("[TRACER] wrote:", out_csv)
    print("[TRACER] wrote:", out_json)
    print()
    print(out[[
        "terrain", "vx", "body_height", "clearance",
        "n", "valid_rate", "bad_rate",
        "mean_cost", "std_cost", "mean_dx", "mean_z_p10",
        "mean_abs_dy_per_dx", "group_label"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
