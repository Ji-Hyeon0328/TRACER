#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


DEFAULT_PROFILES = [
    {
        "name": "safe_v009_n004_s025",
        "goal_distance_ahead": 0.5,
        "vx_far": 0.09,
        "vx_near": 0.04,
        "goal_slow_distance": 0.25,
        "goal_stop_distance": 0.15,
    },
    {
        "name": "mid_v012_n006_s020",
        "goal_distance_ahead": 0.5,
        "vx_far": 0.12,
        "vx_near": 0.06,
        "goal_slow_distance": 0.20,
        "goal_stop_distance": 0.15,
    },
    {
        "name": "fast_v014_n007_s020",
        "goal_distance_ahead": 0.5,
        "vx_far": 0.14,
        "vx_near": 0.07,
        "goal_slow_distance": 0.20,
        "goal_stop_distance": 0.15,
    },
    {
        "name": "fast_v016_n008_s020",
        "goal_distance_ahead": 0.5,
        "vx_far": 0.16,
        "vx_near": 0.08,
        "goal_slow_distance": 0.20,
        "goal_stop_distance": 0.15,
    },
]


def make_features(profile):
    goal = float(profile["goal_distance_ahead"])
    vx_far = float(profile["vx_far"])
    vx_near = float(profile["vx_near"])
    slow = float(profile["goal_slow_distance"])
    stop = float(profile["goal_stop_distance"])

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


def score(model, features):
    weights = model["weights"]
    return sum(float(weights.get(k, 0.0)) * float(features.get(k, 0.0)) for k in model["feature_order"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--profiles-json", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    with open(args.model, "r") as f:
        model = json.load(f)

    if args.profiles_json:
        with open(args.profiles_json, "r") as f:
            profiles = json.load(f)
    else:
        profiles = DEFAULT_PROFILES

    rows = []
    for p in profiles:
        feat = make_features(p)
        pred_score = score(model, feat)
        rows.append({
            "name": p["name"],
            "pred_score": pred_score,
            "profile": p,
            "features": feat,
        })

    rows = sorted(rows, key=lambda r: r["pred_score"], reverse=True)

    print(json.dumps({
        "model": args.model,
        "ranking": rows,
        "selected": rows[0] if rows else None,
    }, indent=2, sort_keys=True))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "model": args.model,
                "ranking": rows,
                "selected": rows[0] if rows else None,
            }, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
