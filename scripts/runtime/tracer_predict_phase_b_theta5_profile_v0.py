#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


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


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def world_features(world_name):
    w = str(world_name or "")
    is_sponge = 1.0 if "sponge" in w else 0.0
    is_flat = 1.0 if ("flat" in w or w == "earth") else 0.0
    is_slope = 1.0 if ("slope" in w and "downslope" not in w) else 0.0
    is_downslope = 1.0 if "downslope" in w else 0.0
    slope_sign = -1.0 if is_downslope else (1.0 if is_slope else 0.0)
    return {
        "world_is_earth": 1.0 if w == "earth" else 0.0,
        "world_is_sponge": is_sponge,
        "world_is_flat": is_flat,
        "world_is_slope": is_slope,
        "world_is_downslope": is_downslope,
        "world_slope_sign": slope_sign,
    }


def make_features(model, risk, world_name, goal_distance, stop_distance):
    wf = world_features(world_name)

    values = {
        "goal_distance_ahead": goal_distance,
        "goal_stop_distance": stop_distance,
        "ram_mismatch": fget(risk, "ram_mismatch"),
        "ram_uncertainty": fget(risk, "ram_uncertainty"),
        "recent_max_yaw_rate": fget(risk, "recent_max_yaw_rate"),
        "recent_slip_score": fget(risk, "recent_slip_score"),
        "body_stability_score": fget(risk, "body_stability_score", 0.9),
        "yaw_p95_abs_rate": fget(risk, "yaw_p95_abs_rate"),
        "yaw_mean_abs_rate": fget(risk, "yaw_mean_abs_rate"),
        "yaw_saturation_fraction": fget(risk, "yaw_saturation_fraction"),
        **wf,
    }

    return [values.get(k, 0.0) for k in model["feature_names"]]


def matvec(W, x):
    return [sum(wj * xj for wj, xj in zip(row, x)) for row in W]


def project(profile, risk, world_name):
    reasons = []

    mismatch = fget(risk, "ram_mismatch")
    unc = fget(risk, "ram_uncertainty")
    yaw = fget(risk, "recent_max_yaw_rate")
    slip = fget(risk, "recent_slip_score")
    stab = fget(risk, "body_stability_score", 1.0)
    sat = fget(risk, "yaw_saturation_fraction")
    w = str(world_name or "")

    if stab < 0.60 or mismatch > 0.60 or unc > 0.60 or slip > 0.50:
        profile["vx_far"] = min(profile["vx_far"], 0.09)
        profile["vx_near"] = min(profile["vx_near"], 0.04)
        profile["goal_slow_distance"] = max(profile["goal_slow_distance"], 0.25)
        reasons.append("hard-conservative projection")

    elif yaw > 0.28 or sat > 0.20:
        profile["vx_far"] = min(profile["vx_far"], 0.12)
        profile["vx_near"] = min(profile["vx_near"], 0.06)
        profile["goal_slow_distance"] = max(profile["goal_slow_distance"], 0.20)
        reasons.append("yaw/saturation projection")

    elif yaw > 0.18 or mismatch > 0.30 or unc > 0.30 or slip > 0.25:
        profile["vx_far"] = min(profile["vx_far"], 0.14)
        profile["vx_near"] = min(profile["vx_near"], 0.07)
        profile["goal_slow_distance"] = max(profile["goal_slow_distance"], 0.20)
        reasons.append("moderate-risk projection")

    # Terrain sanity projection, intentionally mild.
    if "sponge" in w:
        profile["body_height"] = clamp(profile["body_height"], 0.30, 0.36)
        profile["swing_clearance"] = clamp(profile["swing_clearance"], 0.03, 0.09)
        reasons.append("sponge terrain bounds")
    else:
        profile["body_height"] = clamp(profile["body_height"], 0.30, 0.35)
        profile["swing_clearance"] = clamp(profile["swing_clearance"], 0.03, 0.08)

    if not reasons:
        reasons.append("no projection")

    return profile, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--risk-state-json", required=True)
    ap.add_argument("--world-name", default="earth")
    ap.add_argument("--goal-distance-ahead", type=float, default=0.5)
    ap.add_argument("--goal-stop-distance", type=float, default=0.15)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    with open(args.model, "r") as f:
        model = json.load(f)

    with open(args.risk_state_json, "r") as f:
        risk = json.load(f)

    x_raw = make_features(
        model,
        risk,
        args.world_name,
        args.goal_distance_ahead,
        args.goal_stop_distance,
    )

    x_norm = [
        (x_raw[i] - model["x_mean"][i]) / model["x_std"][i]
        for i in range(len(x_raw))
    ]

    xb = [1.0] + x_norm
    y_norm = matvec(model["weights"], xb)

    raw = {}
    profile = {}

    for i, t in enumerate(model["target_names"]):
        val = y_norm[i] * model["y_std"][i] + model["y_mean"][i]
        lo, hi = model["output_limits"][t]
        raw[t] = val
        profile[t] = clamp(val, lo, hi)

    profile["name"] = "theta5_regressed_continuous_v0"
    profile["goal_distance_ahead"] = args.goal_distance_ahead
    profile["goal_stop_distance"] = args.goal_stop_distance
    profile["world_name"] = args.world_name

    projected = dict(profile)
    projected, reasons = project(projected, risk, args.world_name)

    out = {
        "model": args.model,
        "model_type": model.get("model_type"),
        "risk_state_json": args.risk_state_json,
        "risk_state": risk,
        "world_name": args.world_name,
        "raw_prediction": raw,
        "profile": projected,
        "guard_reasons": reasons,
    }

    text = json.dumps(out, indent=2, sort_keys=True)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            f.write(text + "\n")

    print(text)


if __name__ == "__main__":
    main()
