#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path


def ff(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def bb(x):
    return str(x).lower() == "true"


def classify(row):
    reached = bb(row.get("reached"))
    min_d = ff(row.get("min_rel_dist"), 999.0)
    final = ff(row.get("final_rel_dist"), 999.0)
    dx = ff(row.get("odom_x_delta"))
    yaw = ff(row.get("max_yaw"))

    approach_success = min_d <= 0.15
    stable_reached = reached and final <= 0.30 and abs(dx) <= 1.00 and yaw <= 0.30
    drift_after_approach = approach_success and final > 0.30

    return {
        "approach_success": approach_success,
        "stable_reached": stable_reached,
        "drift_after_approach": drift_after_approach,
        "yaw_saturated": yaw >= 0.299,
    }


def score(row, labels):
    min_d = ff(row.get("min_rel_dist"), 999.0)
    final = ff(row.get("final_rel_dist"), 999.0)
    progress = ff(row.get("progress"))
    dx = ff(row.get("odom_x_delta"))
    yaw = ff(row.get("max_yaw"))
    vx = ff(row.get("vx_far"))

    s = 0.0
    s += 150.0 if labels["stable_reached"] else 0.0
    s += 60.0 if labels["approach_success"] else 0.0
    s -= 50.0 if labels["drift_after_approach"] else 0.0
    s += 40.0 * max(progress, 0.0)
    s -= 30.0 * min_d
    s -= 20.0 * final
    s -= 30.0 * max(abs(dx) - 1.0, 0.0)
    s -= 30.0 * max(yaw - 0.20, 0.0)
    s += 2.0 * vx
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", nargs="+", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    args = ap.parse_args()

    examples = []

    for csv_path in args.summary_csv:
        with open(csv_path, "r") as f:
            rows = list(csv.DictReader(f))

        for r in rows:
            labels = classify(r)
            outcome_score = score(r, labels)

            ex = {
                "dataset_type": "phase_b_sponge_probe_outcome_v0",
                "source_csv": csv_path,
                "case_name": r.get("case_name"),
                "world_name": r.get("world"),
                "action": {
                    "vx_far": ff(r.get("vx_far")),
                    "vx_near": ff(r.get("vx_near")),
                    "goal_slow_distance": ff(r.get("slow_distance")),
                    "body_height": ff(r.get("body_height")),
                    "swing_clearance": ff(r.get("swing_clearance")),
                },
                "outcome": {
                    "reached_stop_distance": bb(r.get("reached")),
                    "min_rel_dist": ff(r.get("min_rel_dist")),
                    "final_rel_dist": ff(r.get("final_rel_dist")),
                    "progress_initial_minus_min": ff(r.get("progress")),
                    "odom_x_delta": ff(r.get("odom_x_delta")),
                    "max_abs_mpc_yaw_rate": ff(r.get("max_yaw")),
                },
                "labels": labels,
                "outcome_score": outcome_score,
                "paths": {
                    "run_dir": r.get("run_dir"),
                    "summary_json": r.get("summary_json"),
                },
            }
            examples.append(ex)

    examples.sort(key=lambda x: x["outcome_score"], reverse=True)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_jsonl, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, sort_keys=True) + "\n")

    summary = {
        "num_examples": len(examples),
        "num_stable_reached": sum(1 for e in examples if e["labels"]["stable_reached"]),
        "num_approach_success": sum(1 for e in examples if e["labels"]["approach_success"]),
        "num_drift_after_approach": sum(1 for e in examples if e["labels"]["drift_after_approach"]),
        "best_examples": examples[:5],
        "out_jsonl": args.out_jsonl,
        "source_csvs": args.summary_csv,
    }

    with open(args.out_summary_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
