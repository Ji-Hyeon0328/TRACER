#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="data/rollout_metrics/meta_gait_v0_benchmark_selection.csv")
    ap.add_argument("--rules", default="configs/highlevel_policy/tracer_v0_terrain_gms_rules.yaml")
    ap.add_argument("--out", default="data/rollout_metrics/meta_gait_v0_benchmark_gms_targets.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.selection).open()))
    rules = yaml.safe_load(Path(args.rules).read_text())

    label_to_gms = rules["label_to_gms"]
    terrain_overrides = rules.get("terrain_overrides", {})

    out_rows = []
    for row in rows:
        name = row["name"]
        label = row["label"]

        base_rule = dict(label_to_gms[label])
        override = terrain_overrides.get(name, {})
        semantic_mode = override.get("semantic_mode", base_rule["semantic_mode"])

        out = dict(row)
        out["semantic_mode"] = semantic_mode
        out["risk_level"] = base_rule["risk_level"]
        out["vx_scale"] = base_rule["vx_scale"]
        out["body_height_delta"] = base_rule["body_height_delta"]
        out["clearance_delta"] = base_rule["clearance_delta"]
        out["impedance_scale"] = base_rule["impedance_scale"]
        out["recovery_needed"] = base_rule["recovery_needed"]
        out["gms_note"] = base_rule["note"]
        out_rows.append(out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "benchmark_group",
        "name",
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
        "gms_note",
        "log_path",
    ]

    with out_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"wrote {out_path} with {len(out_rows)} rows")
    for row in out_rows:
        print(
            f"{row['name']:<38s} | "
            f"label={row['label']:<24s} | "
            f"mode={row['semantic_mode']:<32s} | "
            f"risk={row['risk_level']}"
        )


if __name__ == "__main__":
    main()
