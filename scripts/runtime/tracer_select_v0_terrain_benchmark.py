#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


PRIORITY = {
    "stable": [
        "solid_even_forward",
        "solid_downslope5_forward",
        "rough_bumps_start020_forward",
    ],
    "safe_no_progress": [
        "solid_upslope5_forward",
    ],
    "borderline": [
        "rough_bumps_start018_forward",
    ],
    "unstable_slip_or_drift": [
        "icy_slippery_even_forward",
        "sponge_like_even_forward",
        "icy_slippery_downslope5_forward",
        "sponge_like_downslope5_forward",
    ],
    "failure": [
        "mud_slippery_even_forward",
        "icy_slippery_upslope5_forward",
        "mud_slippery_upslope5_forward",
        "sponge_like_upslope5_forward",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/rollout_metrics/meta_gait_eval_matrix_v0.csv")
    ap.add_argument("--out", default="data/rollout_metrics/meta_gait_v0_benchmark_selection.csv")
    args = ap.parse_args()

    matrix = Path(args.matrix)
    rows = list(csv.DictReader(matrix.open()))

    by_name = {r["name"]: r for r in rows}

    selected = []
    for group, names in PRIORITY.items():
        for name in names:
            row = by_name.get(name)
            if row is None:
                print(f"[WARN] missing {name}")
                continue
            row = dict(row)
            row["benchmark_group"] = group
            selected.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "benchmark_group",
        "name",
        "reward_mean",
        "directional_dx",
        "final_dy",
        "abs_final_dy",
        "done_count",
        "mean_theta0",
        "label",
        "log_path",
    ]
    with out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"wrote {out} with {len(selected)} rows")
    for row in selected:
        print(
            f"{row['benchmark_group']:>22s} | {row['name']:<38s} | "
            f"dx={row.get('directional_dx')} "
            f"dy={row.get('abs_final_dy')} "
            f"done={row.get('done_count')} "
            f"label={row.get('label')}"
        )


if __name__ == "__main__":
    main()
