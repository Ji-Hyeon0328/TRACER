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
        "risk_level": "safe",
    },
    {
        "name": "mid_v012_n006_s020",
        "goal_distance_ahead": 0.5,
        "vx_far": 0.12,
        "vx_near": 0.06,
        "goal_slow_distance": 0.20,
        "goal_stop_distance": 0.15,
        "risk_level": "medium",
    },
    {
        "name": "fast_v014_n007_s020",
        "goal_distance_ahead": 0.5,
        "vx_far": 0.14,
        "vx_near": 0.07,
        "goal_slow_distance": 0.20,
        "goal_stop_distance": 0.15,
        "risk_level": "fast",
    },
    {
        "name": "fast_v016_n008_s020",
        "goal_distance_ahead": 0.5,
        "vx_far": 0.16,
        "vx_near": 0.08,
        "goal_slow_distance": 0.20,
        "goal_stop_distance": 0.15,
        "risk_level": "aggressive",
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


def model_score(model, profile):
    features = make_features(profile)
    return sum(
        float(model["weights"].get(k, 0.0)) * float(features.get(k, 0.0))
        for k in model["feature_order"]
    )


def load_risk_state(path):
    if not path:
        return {
            "ram_mismatch": 0.0,
            "ram_uncertainty": 0.0,
            "recent_max_yaw_rate": 0.0,
            "recent_slip_score": 0.0,
            "body_stability_score": 1.0,
        }

    with open(path, "r") as f:
        return json.load(f)


def allowed_by_guard(profile, risk):
    mismatch = float(risk.get("ram_mismatch", 0.0))
    uncertainty = float(risk.get("ram_uncertainty", 0.0))
    yaw = float(risk.get("recent_max_yaw_rate", 0.0))
    slip = float(risk.get("recent_slip_score", 0.0))
    stability = float(risk.get("body_stability_score", 1.0))

    risk_level = profile.get("risk_level", "medium")

    # Hard safety floor.
    if stability < 0.55:
        return risk_level == "safe", "body stability too low"

    if mismatch > 0.60 or uncertainty > 0.60 or slip > 0.50:
        return risk_level in ("safe",), "high mismatch/uncertainty/slip"

    if yaw > 0.28:
        return risk_level in ("safe", "medium"), "recent yaw saturation"

    if mismatch > 0.35 or uncertainty > 0.35 or slip > 0.25:
        return risk_level in ("safe", "medium"), "moderate mismatch/uncertainty/slip"

    if yaw > 0.18:
        return risk_level in ("safe", "medium", "fast"), "moderate yaw activity"

    # Low-risk region allows aggressive profile.
    return True, "allowed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--risk-state-json", default="")
    ap.add_argument("--profiles-json", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    with open(args.model, "r") as f:
        model = json.load(f)

    risk = load_risk_state(args.risk_state_json)

    if args.profiles_json:
        with open(args.profiles_json, "r") as f:
            profiles = json.load(f)
    else:
        profiles = DEFAULT_PROFILES

    ranked = []
    for p in profiles:
        score = model_score(model, p)
        allowed, guard_reason = allowed_by_guard(p, risk)
        ranked.append({
            "name": p["name"],
            "score": score,
            "allowed": allowed,
            "guard_reason": guard_reason,
            "profile": p,
            "features": make_features(p),
        })

    ranked = sorted(ranked, key=lambda r: r["score"], reverse=True)
    allowed_ranked = [r for r in ranked if r["allowed"]]

    selected = allowed_ranked[0] if allowed_ranked else ranked[-1]

    result = {
        "model": args.model,
        "risk_state": risk,
        "ranking": ranked,
        "selected": selected,
        "selection_mode": "learned_selector_plus_risk_guard_v0",
    }

    print(json.dumps(result, indent=2, sort_keys=True))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
