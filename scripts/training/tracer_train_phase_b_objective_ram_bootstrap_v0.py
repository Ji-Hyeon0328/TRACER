#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


FEATURE_NAMES = [
    "world_is_earth",
    "world_is_sponge",
    "world_is_flat",
    "world_is_slope",
    "vx_far",
    "vx_near",
    "goal_slow_distance",
    "body_height",
    "swing_clearance",
]


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


def make_features(ex):
    world = ex.get("world_name") or ex.get("obs", {}).get("terrain", {}).get("world_name", "")
    action = ex.get("obs", {}).get("candidate_action", {})

    wf = world_features(world)

    values = {
        **wf,
        "vx_far": ff(action.get("vx_far")),
        "vx_near": ff(action.get("vx_near")),
        "goal_slow_distance": ff(action.get("goal_slow_distance")),
        "body_height": ff(action.get("body_height")),
        "swing_clearance": ff(action.get("swing_clearance")),
    }

    return [values[k] for k in FEATURE_NAMES]


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def std(xs):
    if len(xs) <= 1:
        return 1.0
    m = mean(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    s = math.sqrt(max(v, 1e-12))
    return s if s > 1e-9 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-jsonl", required=True)
    ap.add_argument("--out-model", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.dataset_jsonl) if line.strip()]
    if not rows:
        raise RuntimeError("empty dataset")

    X = [make_features(r) for r in rows]
    d = len(FEATURE_NAMES)

    x_mean = [mean([x[i] for x in X]) for i in range(d)]
    x_std = [std([x[i] for x in X]) for i in range(d)]

    exemplars = []

    for r, x in zip(rows, X):
        obj = r["objective_target"]
        ram = r["ram_target"]
        action = r["obs"]["candidate_action"]

        exemplars.append({
            "world_name": r.get("world_name"),
            "case_name": r.get("case_name"),
            "source_type": r.get("source_type"),
            "feature_raw": x,
            "feature_norm": [(x[i] - x_mean[i]) / x_std[i] for i in range(d)],
            "action": action,
            "objective_target": {
                "beta_velocity": ff(obj.get("beta_velocity")),
                "beta_stability": ff(obj.get("beta_stability")),
                "beta_energy": ff(obj.get("beta_energy")),
                "semantic": obj.get("semantic"),
                "recovery_needed": bool(obj.get("recovery_needed")),
            },
            "ram_target": {
                "future_risk": ff(ram.get("future_risk")),
                "future_invalid": bool(ram.get("future_invalid")),
                "recovery_needed": bool(ram.get("recovery_needed")),
                "approach_success": bool(ram.get("approach_success")),
                "stable_reached": bool(ram.get("stable_reached")),
                "drift_after_approach": bool(ram.get("drift_after_approach")),
                "yaw_saturated": bool(ram.get("yaw_saturated")),
                "final_rel_dist": ff(ram.get("final_rel_dist")),
                "min_rel_dist": ff(ram.get("min_rel_dist")),
                "abs_odom_x_delta": ff(ram.get("abs_odom_x_delta")),
                "max_abs_yaw_rate": ff(ram.get("max_abs_yaw_rate")),
            },
        })

    semantic_counts = {}
    for e in exemplars:
        sem = e["objective_target"]["semantic"]
        semantic_counts[sem] = semantic_counts.get(sem, 0) + 1

    model = {
        "model_type": "phase_b_objective_ram_bootstrap_knn_v0",
        "feature_names": FEATURE_NAMES,
        "x_mean": x_mean,
        "x_std": x_std,
        "num_examples": len(exemplars),
        "semantic_counts": semantic_counts,
        "exemplars": exemplars,
        "dataset_jsonl": args.dataset_jsonl,
        "note": "Small bootstrap KNN/exemplar predictor for Objective Selector and RAM targets. Intended as a runtime scaffold before larger preference/RAM training.",
    }

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_model, "w") as f:
        json.dump(model, f, indent=2, sort_keys=True)

    print(json.dumps({
        "out_model": args.out_model,
        "num_examples": len(exemplars),
        "semantic_counts": semantic_counts,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
