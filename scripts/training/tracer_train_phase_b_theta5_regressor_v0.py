#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


TARGETS = [
    "vx_far",
    "vx_near",
    "goal_slow_distance",
    "body_height",
    "swing_clearance",
]

OUTPUT_LIMITS = {
    "vx_far": [0.06, 0.16],
    "vx_near": [0.03, 0.08],
    "goal_slow_distance": [0.18, 0.30],
    "body_height": [0.30, 0.36],
    "swing_clearance": [0.03, 0.09],
}


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


FEATURE_NAMES = [
    "goal_distance_ahead",
    "goal_stop_distance",
    "ram_mismatch",
    "ram_uncertainty",
    "recent_max_yaw_rate",
    "recent_slip_score",
    "body_stability_score",
    "yaw_p95_abs_rate",
    "yaw_mean_abs_rate",
    "yaw_saturation_fraction",
    "world_is_earth",
    "world_is_sponge",
    "world_is_flat",
    "world_is_slope",
    "world_is_downslope",
    "world_slope_sign",
]


def make_features(ex):
    obs = ex.get("obs", {})
    risk = obs.get("risk_state", {})
    task = obs.get("task", {})

    world_name = task.get("world_name", ex.get("world_name", "earth"))
    wf = world_features(world_name)

    feat = {
        "goal_distance_ahead": fget(task, "goal_distance_ahead", 0.5),
        "goal_stop_distance": fget(task, "goal_stop_distance", 0.15),
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

    return [feat[k] for k in FEATURE_NAMES]


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def std(xs):
    if len(xs) < 2:
        return 1.0
    m = mean(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    s = math.sqrt(max(v, 1e-12))
    return s if s > 1e-9 else 1.0


def matvec(W, x):
    # W shape: targets x features
    return [sum(wj * xj for wj, xj in zip(row, x)) for row in W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-jsonl", required=True)
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--l2", type=float, default=0.0001)
    args = ap.parse_args()

    examples = load_jsonl(args.dataset_jsonl)
    if not examples:
        raise RuntimeError("empty dataset")

    X_raw = [make_features(ex) for ex in examples]
    Y_raw = []

    for ex in examples:
        a = ex.get("action", {})
        y = []
        for t in TARGETS:
            lo, hi = OUTPUT_LIMITS[t]
            y.append(clamp(fget(a, t), lo, hi))
        Y_raw.append(y)

    n = len(X_raw)
    d = len(FEATURE_NAMES)
    k = len(TARGETS)

    x_mean = [mean([x[j] for x in X_raw]) for j in range(d)]
    x_std = [std([x[j] for x in X_raw]) for j in range(d)]

    y_mean = [mean([y[j] for y in Y_raw]) for j in range(k)]
    y_std = [std([y[j] for y in Y_raw]) for j in range(k)]

    X = []
    for x in X_raw:
        X.append([(x[j] - x_mean[j]) / x_std[j] for j in range(d)])

    Y = []
    for y in Y_raw:
        Y.append([(y[j] - y_mean[j]) / y_std[j] for j in range(k)])

    # bias + normalized features
    Xb = [[1.0] + x for x in X]
    W = [[0.0 for _ in range(d + 1)] for _ in range(k)]

    for epoch in range(1, args.epochs + 1):
        grad = [[0.0 for _ in range(d + 1)] for _ in range(k)]
        loss = 0.0

        for xb, y in zip(Xb, Y):
            pred = matvec(W, xb)
            for ti in range(k):
                err = pred[ti] - y[ti]
                loss += err * err
                for j in range(d + 1):
                    grad[ti][j] += err * xb[j]

        for ti in range(k):
            for j in range(d + 1):
                reg = args.l2 * W[ti][j] if j > 0 else 0.0
                W[ti][j] -= args.lr * ((2.0 / n) * grad[ti][j] + reg)

        if epoch == 1 or epoch % 500 == 0 or epoch == args.epochs:
            rmse = math.sqrt(loss / max(n * k, 1))
            print(f"[train] epoch={epoch} normalized_rmse={rmse:.6f}")

    train_predictions = []
    sq = [0.0 for _ in range(k)]

    for ex, xb, y_true_raw in zip(examples, Xb, Y_raw):
        pred_norm = matvec(W, xb)
        pred_raw = {}
        true_raw = {}

        for ti, t in enumerate(TARGETS):
            lo, hi = OUTPUT_LIMITS[t]
            val = pred_norm[ti] * y_std[ti] + y_mean[ti]
            val = clamp(val, lo, hi)
            pred_raw[t] = val
            true_raw[t] = y_true_raw[ti]
            sq[ti] += (val - y_true_raw[ti]) ** 2

        train_predictions.append({
            "world_name": ex.get("world_name", ex.get("obs", {}).get("task", {}).get("world_name", "")),
            "risk_name": ex.get("risk_name", ""),
            "teacher_case_name": ex.get("teacher_case_name", ""),
            "prediction": pred_raw,
            "target": true_raw,
        })

    rmse_by_target = {
        t: math.sqrt(sq[i] / max(n, 1))
        for i, t in enumerate(TARGETS)
    }

    model = {
        "model_type": "phase_b_theta5_linear_regressor_v0",
        "feature_names": FEATURE_NAMES,
        "target_names": TARGETS,
        "output_limits": OUTPUT_LIMITS,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "weights": W,
        "num_examples": n,
        "train_rmse_by_target": rmse_by_target,
        "train_predictions": train_predictions,
        "dataset_jsonl": args.dataset_jsonl,
        "note": "Bootstrap supervised theta5 regressor from HC sweep teacher dataset. World features are included because terrain-specific height/clearance preferences differ.",
    }

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_model, "w") as f:
        json.dump(model, f, indent=2, sort_keys=True)

    print(json.dumps({
        "num_examples": n,
        "out_model": args.out_model,
        "train_rmse_by_target": rmse_by_target,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
