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


def b(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in ("true", "1", "yes")


def score_row(r):
    reached = b(r.get("reached"))
    progress = ff(r.get("progress"))
    odom = ff(r.get("odom_x_delta"))
    min_dist = ff(r.get("min_rel_dist"), 999.0)
    yaw = ff(r.get("max_yaw"), 999.0)
    vx = ff(r.get("vx_far"))

    score = 0.0
    score += 100.0 if reached else 0.0
    score += 40.0 * max(progress, 0.0)
    score += 8.0 * max(odom, 0.0)
    score += 3.0 * max(vx, 0.0)
    score -= 20.0 * max(min_dist, 0.0)
    score -= 15.0 * max(yaw - 0.18, 0.0)
    score -= 30.0 * max(yaw - 0.28, 0.0)
    return score


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def infer_world(csv_path, row):
    # Prefer explicit world in summary json if available.
    sp = row.get("summary_json", "")
    if sp and Path(sp).is_file():
        try:
            with open(sp, "r") as f:
                s = json.load(f)
            w = s.get("world_name")
            if w:
                return str(w)
        except Exception:
            pass

    # Terrain-set layout: .../phase_b_theta_hc_sweep_terrain_set_xxx/<world>/hc_sweep_summary.csv
    parent = Path(csv_path).parent.name
    if parent and parent not in ("artifacts",):
        return parent

    return "unknown"


def load_csv(path):
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    grouped = defaultdict(list)

    for csv_s in args.summary_csv:
        csv_path = Path(csv_s)
        if not csv_path.is_file():
            print(f"[WARN] missing csv: {csv_path}")
            continue

        rows = load_csv(csv_path)

        for r in rows:
            world = infer_world(csv_path, r)
            case = r.get("case_name", "")
            risk = r.get("risk_name", "")
            h = ff(r.get("body_height"))
            c = ff(r.get("swing_clearance"))

            key = (world, risk, case, h, c)
            item = dict(r)
            item["_source_csv"] = str(csv_path)
            item["_score"] = score_row(r)
            item["_world"] = world
            grouped[key].append(item)

    results = []

    for (world, risk, case, h, c), rows in grouped.items():
        scores = [ff(r["_score"]) for r in rows]
        yaws = [ff(r.get("max_yaw")) for r in rows]
        vxs = [ff(r.get("vx_far")) for r in rows]
        progresses = [ff(r.get("progress")) for r in rows]
        min_dists = [ff(r.get("min_rel_dist")) for r in rows]
        reached = [b(r.get("reached")) for r in rows]

        results.append({
            "world": world,
            "risk_name": risk,
            "case_name": case,
            "body_height": h,
            "swing_clearance": c,
            "n": len(rows),
            "score_mean": mean(scores),
            "score_std": stdev(scores),
            "reached_rate": sum(1 for x in reached if x) / max(len(reached), 1),
            "avg_yaw": mean(yaws),
            "std_yaw": stdev(yaws),
            "avg_vx_far": mean(vxs),
            "avg_progress": mean(progresses),
            "avg_min_rel_dist": mean(min_dists),
            "source_csvs": sorted(set(r["_source_csv"] for r in rows)),
        })

    results.sort(
        key=lambda x: (
            x["world"],
            x["risk_name"],
            -x["score_mean"],
            x["score_std"],
        )
    )

    best_by_world_risk = {}
    for r in results:
        key = f"{r['world']}::{r['risk_name']}"
        if key not in best_by_world_risk:
            best_by_world_risk[key] = r

    out = {
        "num_groups": len(results),
        "best_by_world_risk": best_by_world_risk,
        "ranked_groups": results,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_json, "w") as fp:
        json.dump(out, fp, indent=2, sort_keys=True)

    with open(args.out_csv, "w", newline="") as fp:
        fieldnames = [
            "world",
            "risk_name",
            "case_name",
            "body_height",
            "swing_clearance",
            "n",
            "score_mean",
            "score_std",
            "reached_rate",
            "avg_yaw",
            "std_yaw",
            "avg_vx_far",
            "avg_progress",
            "avg_min_rel_dist",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    print(json.dumps({
        "num_groups": len(results),
        "best_by_world_risk": best_by_world_risk,
        "out_json": args.out_json,
        "out_csv": args.out_csv,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
