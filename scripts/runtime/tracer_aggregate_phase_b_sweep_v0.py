#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


def fget(d, k, default=float("nan")):
    try:
        v = d.get(k, default)
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def bget(d, k, default=False):
    return bool(d.get(k, default))


def score_episode(s):
    reached = bget(s, "reached_stop_distance")
    progress = fget(s, "progress_initial_minus_min", 0.0)
    odom_dx = fget(s, "odom_x_delta", 0.0)
    min_dist = fget(s, "min_rel_dist", 999.0)
    yaw = fget(s, "max_abs_mpc_yaw_rate", 999.0)
    vx = fget(s, "max_mpc_vx", 0.0)

    score = 0.0
    score += 100.0 if reached else 0.0
    score += 40.0 * max(progress, 0.0)
    score += 10.0 * max(odom_dx, 0.0)
    score += 5.0 * vx
    score -= 20.0 * max(min_dist, 0.0)
    score -= 3.0 * max(yaw - 0.3, 0.0)
    return score


def same_task(a, b):
    return (
        abs(fget(a, "goal_distance_ahead") - fget(b, "goal_distance_ahead")) < 1e-9
        and abs(fget(a, "goal_stop_distance") - fget(b, "goal_stop_distance")) < 1e-9
        and a.get("world_name", "") == b.get("world_name", "")
    )


def preference(a, b):
    ar = bget(a, "reached_stop_distance")
    br = bget(b, "reached_stop_distance")

    if ar and not br:
        return 1
    if br and not ar:
        return -1

    ascore = fget(a, "phase_b_score")
    bscore = fget(b, "phase_b_score")

    if abs(ascore - bscore) < 2.0:
        return 0
    return 1 if ascore > bscore else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-prefs", required=True)
    args = ap.parse_args()

    root = Path(args.sweep_root)
    paths = sorted(root.glob("**/phase_b_summary.json"))

    rows = []
    for p in paths:
        try:
            with open(p, "r") as f:
                s = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")
            continue

        s["summary_path"] = str(p)
        s["phase_b_score"] = score_episode(s)
        rows.append(s)

    rows = sorted(rows, key=lambda r: fget(r, "phase_b_score"), reverse=True)

    fields = [
        "phase_b_score",
        "reached_stop_distance",
        "world_name",
        "goal_distance_ahead",
        "vx_far",
        "vx_near",
        "goal_slow_distance",
        "goal_stop_distance",
        "duration_sec",
        "initial_rel_dist",
        "final_rel_dist",
        "min_rel_dist",
        "progress_initial_minus_min",
        "odom_x_delta",
        "max_mpc_vx",
        "max_abs_mpc_yaw_rate",
        "num_rows",
        "num_valid_goal_rows",
        "run_dir",
        "summary_path",
    ]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(args.out_json, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)

    pref_rows = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if not same_task(a, b):
                continue

            pref = preference(a, b)
            if pref == 0:
                continue

            if pref > 0:
                good, bad = a, b
            else:
                good, bad = b, a

            pref_rows.append({
                "preferred_summary": good["summary_path"],
                "rejected_summary": bad["summary_path"],
                "preferred_score": good["phase_b_score"],
                "rejected_score": bad["phase_b_score"],
                "goal_distance_ahead": good.get("goal_distance_ahead"),
                "world_name": good.get("world_name"),
                "reason": "reached/progress/time-risk ranking",
            })

    pref_fields = [
        "preferred_summary",
        "rejected_summary",
        "preferred_score",
        "rejected_score",
        "goal_distance_ahead",
        "world_name",
        "reason",
    ]

    with open(args.out_prefs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pref_fields)
        w.writeheader()
        for r in pref_rows:
            w.writerow(r)

    print(json.dumps({
        "sweep_root": str(root),
        "num_episodes": len(rows),
        "num_preference_pairs": len(pref_rows),
        "out_csv": args.out_csv,
        "out_json": args.out_json,
        "out_prefs": args.out_prefs,
        "best": rows[0] if rows else None,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
