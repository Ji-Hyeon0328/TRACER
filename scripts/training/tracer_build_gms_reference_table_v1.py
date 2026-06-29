#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def pick_best(group: pd.DataFrame, label_filter=None, top_k=3):
    g = group.copy()
    if label_filter is not None:
        g = g[g["suitability_label"].isin(label_filter)].copy()
    if len(g) == 0:
        return []
    g = g.sort_values("cost")
    out = []
    for _, r in g.head(top_k).iterrows():
        out.append({
            "episode_id": str(r["episode_id"]),
            "preset": str(r.get("preset", "")),
            "label": str(r["suitability_label"]),
            "cost": float(r["cost"]),
            "vx": float(r["vx"]),
            "body_height": float(r["body_height"]),
            "clearance": float(r["clearance"]),
            "dx": float(r["dx"]),
            "z_p10": float(r["z_p10"]),
            "abs_dy_per_dx": float(r["abs_dy_per_dx"]),
            "roll_abs_max": float(r["roll_abs_max"]) if pd.notna(r["roll_abs_max"]) else None,
            "pitch_abs_max": float(r["pitch_abs_max"]) if pd.notna(r["pitch_abs_max"]) else None,
            "yaw_delta": float(r["yaw_delta"]) if pd.notna(r["yaw_delta"]) else None,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="data/rollout_metrics/gms_suitability_dataset_v1.csv")
    ap.add_argument("--out-json", default="data/rollout_metrics/gms_reference_table_v1.json")
    ap.add_argument("--out-csv", default="data/rollout_metrics/gms_reference_table_v1.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    usable = df[
        (df["valid_data"] == True)
        & (df["has_attitude"] == True)
    ].copy()

    rows = []
    table = {
        "version": "gms_reference_table_v1",
        "source": args.in_csv,
        "terrains": {},
    }

    for terrain, g in usable.groupby("terrain"):
        preferred = pick_best(
            g,
            label_filter=["preferred_fast_flat", "preferred_balanced_flat"],
            top_k=5,
        )
        conservative = pick_best(
            g,
            label_filter=["acceptable_slow_flat", "preferred_balanced_flat"],
            top_k=5,
        )
        avoid = pick_best(
            g,
            label_filter=[
                "negative_lateral_drift",
                "negative_low_height",
                "negative_unstable_attitude",
                "variable_forward_with_lateral_drift",
            ],
            top_k=10,
        )

        table["terrains"][terrain] = {
            "preferred": preferred,
            "conservative": conservative,
            "avoid": avoid,
        }

        for bucket, items in table["terrains"][terrain].items():
            for rank, item in enumerate(items, start=1):
                rr = {"terrain": terrain, "bucket": bucket, "rank": rank}
                rr.update(item)
                rows.append(rr)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(table, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print("[TRACER] wrote:", out_json)
    print("[TRACER] wrote:", out_csv)
    print(json.dumps(table, indent=2)[:4000])


if __name__ == "__main__":
    main()
