#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def fget(d, k, default=0.0):
    try:
        v = d.get(k, default)
        if v is None or v == "":
            return default
        x = float(v)
        if not math.isfinite(x):
            return default
        return x
    except Exception:
        return default


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def features_from_risk_and_task(risk, goal_distance, stop_distance):
    return {
        "goal_distance_ahead": goal_distance,
        "goal_stop_distance": stop_distance,
        "ram_mismatch": fget(risk, "ram_mismatch"),
        "ram_uncertainty": fget(risk, "ram_uncertainty"),
        "recent_max_yaw_rate": fget(risk, "recent_max_yaw_rate"),
        "recent_slip_score": fget(risk, "recent_slip_score"),
        "body_stability_score": fget(risk, "body_stability_score", 1.0),
        "yaw_p95_abs_rate": fget(risk, "yaw_p95_abs_rate"),
        "yaw_mean_abs_rate": fget(risk, "yaw_mean_abs_rate"),
        "yaw_saturation_fraction": fget(risk, "yaw_saturation_fraction"),
    }


def predict(model, risk, goal_distance, stop_distance):
    feat = features_from_risk_and_task(risk, goal_distance, stop_distance)

    x = []
    for k in model["feature_order"]:
        mu = float(model["feature_mean"].get(k, 0.0))
        sig = float(model["feature_std"].get(k, 1.0))
        if abs(sig) < 1e-12:
            sig = 1.0
        x.append((float(feat.get(k, 0.0)) - mu) / sig)

    raw = {}
    for target in model["target_order"]:
        b = float(model["bias"].get(target, 0.0))
        wdict = model["weights"].get(target, {})
        w = [float(wdict.get(k, 0.0)) for k in model["feature_order"]]
        raw[target] = b + dot(w, x)

    limits = model.get("output_limits", {})
    projected = {}
    for k, v in raw.items():
        lo, hi = limits.get(k, [-1e9, 1e9])
        projected[k] = clamp(v, float(lo), float(hi))

    # Safety projection for out-of-distribution high-risk states.
    mismatch = fget(risk, "ram_mismatch")
    uncertainty = fget(risk, "ram_uncertainty")
    yaw = fget(risk, "recent_max_yaw_rate")
    slip = fget(risk, "recent_slip_score")
    stability = fget(risk, "body_stability_score", 1.0)
    sat_frac = fget(risk, "yaw_saturation_fraction")

    guard_reasons = []

    if stability < 0.60 or mismatch > 0.60 or uncertainty > 0.60 or slip > 0.50:
        projected["vx_far"] = min(projected["vx_far"], 0.09)
        projected["vx_near"] = min(projected["vx_near"], 0.04)
        projected["goal_slow_distance"] = max(projected["goal_slow_distance"], 0.25)
        guard_reasons.append("hard conservative projection")

    elif yaw > 0.28 or sat_frac > 0.20:
        projected["vx_far"] = min(projected["vx_far"], 0.12)
        projected["vx_near"] = min(projected["vx_near"], 0.06)
        projected["goal_slow_distance"] = max(projected["goal_slow_distance"], 0.20)
        guard_reasons.append("yaw/saturation projection")

    elif yaw > 0.18 or mismatch > 0.30 or uncertainty > 0.30 or slip > 0.25:
        projected["vx_far"] = min(projected["vx_far"], 0.14)
        projected["vx_near"] = min(projected["vx_near"], 0.07)
        projected["goal_slow_distance"] = max(projected["goal_slow_distance"], 0.20)
        guard_reasons.append("moderate-risk projection")

    if not guard_reasons:
        guard_reasons.append("no projection")

    profile = {
        "name": "theta_regressed_continuous_v0",
        "goal_distance_ahead": goal_distance,
        "goal_stop_distance": stop_distance,
        "vx_far": projected["vx_far"],
        "vx_near": projected["vx_near"],
        "goal_slow_distance": projected["goal_slow_distance"],
    }

    return {
        "features": feat,
        "raw_prediction": raw,
        "projected_prediction": projected,
        "profile": profile,
        "guard_reasons": guard_reasons,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--risk-state-json", required=True)
    ap.add_argument("--goal-distance-ahead", type=float, default=0.5)
    ap.add_argument("--goal-stop-distance", type=float, default=0.15)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    model = load_json(args.model)
    risk = load_json(args.risk_state_json)

    result = predict(
        model,
        risk,
        goal_distance=args.goal_distance_ahead,
        stop_distance=args.goal_stop_distance,
    )

    result["model"] = args.model
    result["risk_state"] = risk

    print(json.dumps(result, indent=2, sort_keys=True))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
