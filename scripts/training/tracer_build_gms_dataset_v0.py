#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


MATERIALS = [
    "icy_slippery",
    "mud_slippery",
    "sponge_like",
    "solid",
    "rough_bumps",
]


def parse_name(name: str) -> dict:
    material = "unknown"
    for m in MATERIALS:
        if name.startswith(m + "_"):
            material = m
            break

    slope_type = "even"
    slope_deg = 0.0
    m = re.search(r"_(upslope|downslope)([0-9]+)_", name)
    if m:
        slope_type = m.group(1)
        slope_deg = float(m.group(2))

    geometry = "flat"
    rough_start_x = ""
    m = re.search(r"rough_bumps_start([0-9]+)_", name)
    if m:
        geometry = "rough_bumps"
        raw = m.group(1)
        if len(raw) >= 2:
            rough_start_x = float("0." + raw[-2:])
        else:
            rough_start_x = float(raw)

    gait_direction = "forward"
    if name.endswith("_backward") or "_backward_" in name:
        gait_direction = "backward"

    return {
        "material": material,
        "slope_type": slope_type,
        "slope_deg": slope_deg,
        "geometry": geometry,
        "rough_start_x": rough_start_x,
        "gait_direction": gait_direction,
        "is_slippery": int(material in ("icy_slippery", "mud_slippery")),
        "is_soft": int(material == "sponge_like"),
        "is_rough": int(geometry == "rough_bumps"),
        "is_upslope": int(slope_type == "upslope"),
        "is_downslope": int(slope_type == "downslope"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="data/rollout_metrics/meta_gait_v0_benchmark_gms_targets.csv")
    ap.add_argument("--out", default="data/rollout_metrics/meta_gait_v0_gms_dataset.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.targets).open()))
    out_rows = []

    for row in rows:
        features = parse_name(row["name"])
        out = {
            **features,
            **row,
        }
        out_rows.append(out)

    fieldnames = [
        "name",
        "material",
        "slope_type",
        "slope_deg",
        "geometry",
        "rough_start_x",
        "gait_direction",
        "is_slippery",
        "is_soft",
        "is_rough",
        "is_upslope",
        "is_downslope",
        "benchmark_group",
        "label",
        "semantic_mode",
        "risk_level",
        "vx_scale",
        "body_height_delta",
        "clearance_delta",
        "impedance_scale",
        "recovery_needed",
        "reward_mean",
        "directional_dx",
        "final_dy",
        "abs_final_dy",
        "done_count",
        "mean_theta0",
        "log_path",
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"wrote {out_path} with {len(out_rows)} rows")
    for row in out_rows:
        print(
            f"{row['name']:<38s} | "
            f"mat={row['material']:<13s} "
            f"slope={row['slope_type']:<9s}{str(row['slope_deg']):>4s} "
            f"geom={row['geometry']:<11s} "
            f"mode={row['semantic_mode']}"
        )


if __name__ == "__main__":
    main()
