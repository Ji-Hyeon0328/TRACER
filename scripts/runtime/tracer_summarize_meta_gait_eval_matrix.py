#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path


METRICS = [
    "reward_mean",
    "final_dx",
    "directional_dx",
    "final_dy",
    "abs_final_dy",
    "done_count",
    "mean_theta0",
]


def parse_metric(text: str, key: str):
    m = re.search(rf"^\s*{re.escape(key)}:\s*([+-]?[0-9]*\.?[0-9]+)", text, re.MULTILINE)
    if not m:
        return None
    val = m.group(1)
    if key == "done_count":
        return int(float(val))
    return float(val)


def classify(row: dict) -> str:
    dx = row.get("directional_dx")
    dy = row.get("abs_final_dy")
    if dy is None:
        dy = abs(row.get("final_dy", 0.0))
    done = row.get("done_count")
    rew = row.get("reward_mean")

    if dx is None or done is None or rew is None:
        return "missing_metrics"

    if done == 0 and dx > 0.02 and abs(dy) < 0.02 and rew > -1.0:
        return "stable"
    if done == 0 and dx <= 0.0:
        return "safe_no_progress"
    if dx > 0.02 and (done > 0 or rew < -1.0):
        return "unstable_slip_or_drift"
    if dx <= 0.0 and done > 0:
        return "failure"
    return "borderline"


def infer_name(path: Path) -> str:
    name = path.name
    prefix = "eval_active0_"
    suffix = "_legacy_noctx_model19_"
    if name.startswith(prefix) and suffix in name:
        return name[len(prefix): name.index(suffix)]
    return path.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs/tracer_runs")
    ap.add_argument("--out", default="data/rollout_metrics/meta_gait_eval_matrix_v0.csv")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(str(Path(args.log_dir) / "eval_active0_*_legacy_noctx_model19_*.log"))):
        path = Path(f)
        text = path.read_text(errors="replace")
        row = {
            "name": infer_name(path),
            "log_path": str(path),
        }
        for key in METRICS:
            row[key] = parse_metric(text, key)
        if row.get("abs_final_dy") is None and row.get("final_dy") is not None:
            row["abs_final_dy"] = abs(row["final_dy"])
        row["label"] = classify(row)
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["name", *METRICS, "label", "log_path"]
    with out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out} with {len(rows)} rows")
    for row in rows:
        print(
            f"{row['name']}: "
            f"dx={row.get('directional_dx')} "
            f"dy={row.get('abs_final_dy')} "
            f"done={row.get('done_count')} "
            f"rew={row.get('reward_mean')} "
            f"label={row['label']}"
        )


if __name__ == "__main__":
    main()
