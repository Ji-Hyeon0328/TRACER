#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


def ff(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def world_features(world_name):
    w = str(world_name or "")
    return {
        "world_is_earth": 1.0 if w == "earth" else 0.0,
        "world_is_sponge": 1.0 if "sponge" in w else 0.0,
        "world_is_flat": 1.0 if w == "earth" or "flat" in w else 0.0,
        "world_is_slope": 1.0 if "slope" in w else 0.0,
    }


def make_features(model, args):
    wf = world_features(args.world_name)
    values = {
        **wf,
        "vx_far": args.vx_far,
        "vx_near": args.vx_near,
        "goal_slow_distance": args.goal_slow_distance,
        "body_height": args.body_height,
        "swing_clearance": args.swing_clearance,
    }
    return [values.get(k, 0.0) for k in model["feature_names"]]


def dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def weighted_bool(neighbors, key, section):
    num = 0.0
    den = 0.0
    for w, e in neighbors:
        num += w * (1.0 if e[section].get(key) else 0.0)
        den += w
    return (num / max(den, 1e-12)) >= 0.5


def weighted_float(neighbors, key, section):
    num = 0.0
    den = 0.0
    for w, e in neighbors:
        num += w * ff(e[section].get(key))
        den += w
    return num / max(den, 1e-12)


def weighted_semantic(neighbors):
    votes = {}
    for w, e in neighbors:
        sem = e["objective_target"].get("semantic", "unknown")
        votes[sem] = votes.get(sem, 0.0) + w
    return max(votes.items(), key=lambda kv: kv[1])[0]


def normalize_beta(beta):
    z = sum(beta.values())
    if z <= 0:
        return beta
    return {k: v / z for k, v in beta.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--world-name", default="earth")
    ap.add_argument("--vx-far", type=float, required=True)
    ap.add_argument("--vx-near", type=float, required=True)
    ap.add_argument("--goal-slow-distance", type=float, required=True)
    ap.add_argument("--body-height", type=float, required=True)
    ap.add_argument("--swing-clearance", type=float, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    with open(args.model, "r") as f:
        model = json.load(f)

    x_raw = make_features(model, args)
    x_norm = [
        (x_raw[i] - model["x_mean"][i]) / model["x_std"][i]
        for i in range(len(x_raw))
    ]

    scored = []
    for e in model["exemplars"]:
        d2 = dist2(x_norm, e["feature_norm"])
        scored.append((d2, e))

    scored.sort(key=lambda x: x[0])
    selected = scored[: max(1, args.k)]

    neighbors = []
    for d2, e in selected:
        w = 1.0 / (d2 + 1e-6)
        neighbors.append((w, e))

    beta = normalize_beta({
        "beta_velocity": weighted_float(neighbors, "beta_velocity", "objective_target"),
        "beta_stability": weighted_float(neighbors, "beta_stability", "objective_target"),
        "beta_energy": weighted_float(neighbors, "beta_energy", "objective_target"),
    })

    semantic = weighted_semantic(neighbors)

    ram = {
        "future_risk": weighted_float(neighbors, "future_risk", "ram_target"),
        "future_invalid": weighted_bool(neighbors, "future_invalid", "ram_target"),
        "recovery_needed": weighted_bool(neighbors, "recovery_needed", "ram_target"),
        "approach_success": weighted_bool(neighbors, "approach_success", "ram_target"),
        "stable_reached": weighted_bool(neighbors, "stable_reached", "ram_target"),
        "drift_after_approach": weighted_bool(neighbors, "drift_after_approach", "ram_target"),
        "yaw_saturated": weighted_bool(neighbors, "yaw_saturated", "ram_target"),
        "pred_final_rel_dist": weighted_float(neighbors, "final_rel_dist", "ram_target"),
        "pred_min_rel_dist": weighted_float(neighbors, "min_rel_dist", "ram_target"),
        "pred_abs_odom_x_delta": weighted_float(neighbors, "abs_odom_x_delta", "ram_target"),
        "pred_max_abs_yaw_rate": weighted_float(neighbors, "max_abs_yaw_rate", "ram_target"),
    }

    out = {
        "model": args.model,
        "model_type": model.get("model_type"),
        "query": {
            "world_name": args.world_name,
            "action": {
                "vx_far": args.vx_far,
                "vx_near": args.vx_near,
                "goal_slow_distance": args.goal_slow_distance,
                "body_height": args.body_height,
                "swing_clearance": args.swing_clearance,
            },
        },
        "objective_prediction": {
            **beta,
            "semantic": semantic,
            "recovery_needed": ram["recovery_needed"],
        },
        "ram_prediction": ram,
        "neighbors": [
            {
                "rank": i + 1,
                "distance": math.sqrt(max(d2, 0.0)),
                "weight": 1.0 / (d2 + 1e-6),
                "world_name": e.get("world_name"),
                "case_name": e.get("case_name"),
                "source_type": e.get("source_type"),
                "semantic": e["objective_target"].get("semantic"),
                "future_risk": e["ram_target"].get("future_risk"),
                "stable_reached": e["ram_target"].get("stable_reached"),
                "recovery_needed": e["ram_target"].get("recovery_needed"),
                "action": e.get("action"),
            }
            for i, (d2, e) in enumerate(selected)
        ],
    }

    text = json.dumps(out, indent=2, sort_keys=True)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            f.write(text + "\n")

    print(text)


if __name__ == "__main__":
    main()
