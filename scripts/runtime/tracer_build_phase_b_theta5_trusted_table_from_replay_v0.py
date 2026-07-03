#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def ff(x, default=0.0):
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def bb(x):
    return str(x).lower() == "true"


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-table-json", default="configs/phase_b_theta5_teacher_table_v0/best_table.json")
    ap.add_argument("--replay-summary-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--min-repeats", type=int, default=2)
    ap.add_argument("--min-reached-rate", type=float, default=0.80)
    ap.add_argument("--max-avg-final-dist", type=float, default=0.30)
    ap.add_argument("--max-abs-avg-odom-dx", type=float, default=1.00)
    args = ap.parse_args()

    with open(args.teacher_table_json, "r") as f:
        teacher = json.load(f)

    rows = list(csv.DictReader(open(args.replay_summary_csv)))

    groups = defaultdict(list)
    for r in rows:
        groups[r["key"]].append(r)

    trusted = {}
    rejected = {}

    for key, rs in sorted(groups.items()):
        n = len(rs)
        reached_rate = sum(bb(r.get("reached")) for r in rs) / max(n, 1)
        avg_min = mean([ff(r.get("min_rel_dist")) for r in rs])
        avg_final = mean([ff(r.get("final_rel_dist")) for r in rs])
        avg_progress = mean([ff(r.get("progress")) for r in rs])
        avg_dx = mean([ff(r.get("odom_x_delta")) for r in rs])
        avg_yaw = mean([ff(r.get("max_yaw")) for r in rs])

        reasons = []
        if n < args.min_repeats:
            reasons.append(f"n<{args.min_repeats}")
        if reached_rate < args.min_reached_rate:
            reasons.append(f"reached_rate<{args.min_reached_rate}")
        if avg_final > args.max_avg_final_dist:
            reasons.append(f"avg_final>{args.max_avg_final_dist}")
        if abs(avg_dx) > args.max_abs_avg_odom_dx:
            reasons.append(f"abs(avg_odom_dx)>{args.max_abs_avg_odom_dx}")

        stats = {
            "replay_n": n,
            "replay_reached_rate": reached_rate,
            "replay_avg_min_rel_dist": avg_min,
            "replay_avg_final_rel_dist": avg_final,
            "replay_avg_progress": avg_progress,
            "replay_avg_odom_x_delta": avg_dx,
            "replay_avg_max_yaw": avg_yaw,
            "replay_summary_csv": args.replay_summary_csv,
        }

        entry = teacher.get("best_table", {}).get(key)

        if entry is None:
            rejected[key] = {
                "reason": ["missing in teacher table"] + reasons,
                "replay_stats": stats,
            }
            continue

        merged = dict(entry)
        merged["replay_stats"] = stats
        merged["trust_rule"] = {
            "min_repeats": args.min_repeats,
            "min_reached_rate": args.min_reached_rate,
            "max_avg_final_dist": args.max_avg_final_dist,
            "max_abs_avg_odom_dx": args.max_abs_avg_odom_dx,
        }

        if reasons:
            rejected[key] = {
                "reason": reasons,
                "entry": merged,
            }
        else:
            trusted[key] = merged

    out = {
        "table_type": "phase_b_theta5_trusted_teacher_table_v0",
        "source_teacher_table_json": args.teacher_table_json,
        "source_replay_summary_csv": args.replay_summary_csv,
        "trust_rule": {
            "min_repeats": args.min_repeats,
            "min_reached_rate": args.min_reached_rate,
            "max_avg_final_dist": args.max_avg_final_dist,
            "max_abs_avg_odom_dx": args.max_abs_avg_odom_dx,
        },
        "best_table": trusted,
        "trusted_keys": sorted(trusted.keys()),
        "rejected_keys": sorted(rejected.keys()),
        "rejected": rejected,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(json.dumps({
        "out_json": args.out_json,
        "num_trusted": len(trusted),
        "trusted_keys": sorted(trusted.keys()),
        "num_rejected": len(rejected),
        "rejected_keys": sorted(rejected.keys()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
