#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path


def fget(d, k, default=0.0):
    try:
        v = d.get(k, default)
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def bget(d, k, default=False):
    v = d.get(k, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


def load_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def candidate_features(row):
    goal = fget(row, "goal_distance_ahead")
    vx_far = fget(row, "vx_far")
    vx_near = fget(row, "vx_near")
    slow = fget(row, "goal_slow_distance")
    stop = fget(row, "goal_stop_distance")

    # Candidate/meta-action features only.
    # Avoid using outcome metrics here, because at deployment we do not know the future outcome.
    return {
        "bias": 1.0,
        "goal_distance_ahead": goal,
        "vx_far": vx_far,
        "vx_near": vx_near,
        "goal_slow_distance": slow,
        "goal_stop_distance": stop,
        "vx_far_over_goal": vx_far / max(goal, 1e-6),
        "vx_near_over_goal": vx_near / max(goal, 1e-6),
        "slow_over_goal": slow / max(goal, 1e-6),
    }


def outcome_summary(row):
    return {
        "reached_stop_distance": bget(row, "reached_stop_distance"),
        "phase_b_score": fget(row, "phase_b_score"),
        "initial_rel_dist": fget(row, "initial_rel_dist"),
        "final_rel_dist": fget(row, "final_rel_dist"),
        "min_rel_dist": fget(row, "min_rel_dist"),
        "progress_initial_minus_min": fget(row, "progress_initial_minus_min"),
        "odom_x_delta": fget(row, "odom_x_delta"),
        "max_mpc_vx": fget(row, "max_mpc_vx"),
        "max_abs_mpc_yaw_rate": fget(row, "max_abs_mpc_yaw_rate"),
        "num_valid_goal_rows": fget(row, "num_valid_goal_rows"),
    }


def risk_aware_score(row):
    reached = bget(row, "reached_stop_distance")
    progress = fget(row, "progress_initial_minus_min")
    odom_dx = fget(row, "odom_x_delta")
    min_dist = fget(row, "min_rel_dist", 999.0)
    yaw = fget(row, "max_abs_mpc_yaw_rate", 999.0)
    vx = fget(row, "max_mpc_vx")

    score = 0.0
    score += 100.0 if reached else 0.0
    score += 40.0 * max(progress, 0.0)
    score += 8.0 * max(odom_dx, 0.0)
    score += 4.0 * max(vx, 0.0)
    score -= 20.0 * max(min_dist, 0.0)

    # Stronger risk penalty than the initial aggregator.
    # yaw_rate near saturation is allowed, but should lose preference unless it clearly improves completion.
    score -= 15.0 * max(yaw - 0.15, 0.0)
    score -= 30.0 * max(yaw - 0.28, 0.0)

    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-summary-csv", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-candidates-json", required=True)
    args = ap.parse_args()

    rows = load_csv(args.sweep_summary_csv)

    by_task = {}
    for r in rows:
        task_key = (
            r.get("world_name", ""),
            str(fget(r, "goal_distance_ahead")),
            str(fget(r, "goal_stop_distance")),
        )
        r["_risk_aware_score"] = risk_aware_score(r)
        by_task.setdefault(task_key, []).append(r)

    examples = []
    candidates = []

    for r in rows:
        candidates.append({
            "summary_path": r.get("summary_path", ""),
            "run_dir": r.get("run_dir", ""),
            "world_name": r.get("world_name", ""),
            "candidate_features": candidate_features(r),
            "outcome": outcome_summary(r),
            "risk_aware_score": r["_risk_aware_score"],
        })

    for task_key, group in by_task.items():
        group = sorted(group, key=lambda x: x["_risk_aware_score"], reverse=True)

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                good = group[i]
                bad = group[j]

                # Skip ambiguous pairs.
                margin = good["_risk_aware_score"] - bad["_risk_aware_score"]
                if margin < 2.0:
                    continue

                # If both reached and risk/score are nearly identical, skip.
                ex = {
                    "task": {
                        "world_name": good.get("world_name", ""),
                        "goal_distance_ahead": fget(good, "goal_distance_ahead"),
                        "goal_stop_distance": fget(good, "goal_stop_distance"),
                    },
                    "preferred": {
                        "summary_path": good.get("summary_path", ""),
                        "features": candidate_features(good),
                        "outcome": outcome_summary(good),
                        "risk_aware_score": good["_risk_aware_score"],
                    },
                    "rejected": {
                        "summary_path": bad.get("summary_path", ""),
                        "features": candidate_features(bad),
                        "outcome": outcome_summary(bad),
                        "risk_aware_score": bad["_risk_aware_score"],
                    },
                    "label": 1,
                    "margin": margin,
                    "reason": "risk-aware speed preference from Phase-B sweep",
                }
                examples.append(ex)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_jsonl, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, sort_keys=True) + "\n")

    with open(args.out_candidates_json, "w") as f:
        json.dump(candidates, f, indent=2, sort_keys=True)

    print(json.dumps({
        "num_candidates": len(candidates),
        "num_preference_examples": len(examples),
        "out_jsonl": args.out_jsonl,
        "out_candidates_json": args.out_candidates_json,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
