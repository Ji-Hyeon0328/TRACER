#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path


def f(x, default=0.0):
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def load_rows(csv_path):
    with open(csv_path, "r") as fp:
        return list(csv.DictReader(fp))


def load_json(path):
    with open(path, "r") as fp:
        return json.load(fp)


def score_run(rows):
    n = max(len(rows), 1)

    reached = [str(r.get("reached", "")).lower() == "true" for r in rows]
    vx = [f(r.get("vx_far")) for r in rows]
    yaw = [f(r.get("max_yaw")) for r in rows]
    progress = [f(r.get("progress")) for r in rows]
    final_dist = [f(r.get("final_rel_dist")) for r in rows]

    reached_rate = sum(1 for x in reached if x) / n
    avg_vx = sum(vx) / n
    avg_yaw = sum(yaw) / n
    avg_progress = sum(progress) / n
    avg_final_dist = sum(final_dist) / n
    yaw_sat_frac = sum(1 for y in yaw if y >= 0.285) / n

    # Higher is better.
    # Priority:
    # 1. reach reliably
    # 2. reduce sustained/high yaw
    # 3. keep useful forward speed/progress
    score = 0.0
    score += 100.0 * reached_rate
    score += 25.0 * avg_progress
    score += 20.0 * avg_vx
    score -= 45.0 * avg_yaw
    score -= 10.0 * yaw_sat_frac
    score -= 8.0 * avg_final_dist

    return {
        "score": score,
        "num_episodes": n,
        "reached_rate": reached_rate,
        "avg_vx_far": avg_vx,
        "avg_max_yaw": avg_yaw,
        "avg_progress": avg_progress,
        "avg_final_rel_dist": avg_final_dist,
        "yaw_saturation_episode_fraction": yaw_sat_frac,
        "vx_far_values": vx,
        "max_yaw_values": yaw,
        "reached_values": reached,
    }


def infer_theta_model(run_root):
    # The runner injects theta_model into episode summaries.
    for p in sorted(Path(run_root).glob("episode_*/episode/phase_b_summary.json")):
        try:
            s = load_json(p)
            m = s.get("theta_model", "")
            if m:
                return m
        except Exception:
            pass
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    results = []

    for root_s in args.roots:
        root = Path(root_s)
        csv_path = root / "adaptive_theta_summary.csv"
        if not csv_path.is_file():
            print(f"[WARN] missing adaptive theta summary: {csv_path}")
            continue

        rows = load_rows(csv_path)
        metrics = score_run(rows)
        theta_model = infer_theta_model(root)

        result = {
            "run_root": str(root),
            "adaptive_theta_summary_csv": str(csv_path),
            "theta_model": theta_model,
            **metrics,
        }
        results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)

    out = {
        "num_runs": len(results),
        "best_run": results[0] if results else None,
        "ranked_runs": results,
        "scoring_note": (
            "Higher score is better. This heuristic prioritizes reached_rate, "
            "then yaw reduction, then useful speed/progress."
        ),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fp:
        json.dump(out, fp, indent=2, sort_keys=True)

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
