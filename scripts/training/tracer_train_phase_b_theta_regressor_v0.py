#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


FEATURE_ORDER = [
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
]

TARGET_ORDER = [
    "vx_far",
    "vx_near",
    "goal_slow_distance",
]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def features_from_example(ex):
    risk = ex["obs"]["risk_state"]
    task = ex["obs"]["task"]

    return {
        "goal_distance_ahead": fget(task, "goal_distance_ahead", 0.5),
        "goal_stop_distance": fget(task, "goal_stop_distance", 0.15),
        "ram_mismatch": fget(risk, "ram_mismatch"),
        "ram_uncertainty": fget(risk, "ram_uncertainty"),
        "recent_max_yaw_rate": fget(risk, "recent_max_yaw_rate"),
        "recent_slip_score": fget(risk, "recent_slip_score"),
        "body_stability_score": fget(risk, "body_stability_score", 1.0),
        "yaw_p95_abs_rate": fget(risk, "yaw_p95_abs_rate"),
        "yaw_mean_abs_rate": fget(risk, "yaw_mean_abs_rate"),
        "yaw_saturation_fraction": fget(risk, "yaw_saturation_fraction"),
    }


def target_from_example(ex):
    action = ex["action"]
    return {
        "vx_far": fget(action, "vx_far"),
        "vx_near": fget(action, "vx_near"),
        "goal_slow_distance": fget(action, "goal_slow_distance"),
    }


def vec_from_dict(d, order):
    return [float(d.get(k, 0.0)) for k in order]


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def std(xs):
    m = mean(xs)
    v = sum((x - m) ** 2 for x in xs) / max(len(xs), 1)
    return math.sqrt(v) if v > 1e-12 else 1.0


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-jsonl", required=True)
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--l2", type=float, default=1e-4)
    args = ap.parse_args()

    data = load_jsonl(args.dataset_jsonl)
    if not data:
        raise RuntimeError(f"No examples found: {args.dataset_jsonl}")

    X_raw = [vec_from_dict(features_from_example(ex), FEATURE_ORDER) for ex in data]
    Y = [vec_from_dict(target_from_example(ex), TARGET_ORDER) for ex in data]

    feat_mean = []
    feat_std = []
    for j in range(len(FEATURE_ORDER)):
        col = [x[j] for x in X_raw]
        feat_mean.append(mean(col))
        feat_std.append(std(col))

    X = []
    for x in X_raw:
        X.append([(x[j] - feat_mean[j]) / feat_std[j] for j in range(len(FEATURE_ORDER))])

    # Multi-output linear regressor:
    # y_k = bias_k + W_k dot x
    target_mean = []
    for k in range(len(TARGET_ORDER)):
        target_mean.append(mean([y[k] for y in Y]))

    B = target_mean[:]
    W = [[0.0 for _ in FEATURE_ORDER] for _ in TARGET_ORDER]

    n = len(data)

    for epoch in range(args.epochs):
        total_loss = 0.0

        for x, y in zip(X, Y):
            for k in range(len(TARGET_ORDER)):
                pred = B[k] + dot(W[k], x)
                err = pred - y[k]
                total_loss += err * err

                B[k] -= args.lr * err
                for j in range(len(FEATURE_ORDER)):
                    grad = err * x[j] + args.l2 * W[k][j]
                    W[k][j] -= args.lr * grad

        if epoch in (0, args.epochs - 1) or (epoch + 1) % 500 == 0:
            rmse = math.sqrt(total_loss / max(n * len(TARGET_ORDER), 1))
            print(f"[train] epoch={epoch+1} rmse={rmse:.6f}")

    rows = []
    sq = [0.0 for _ in TARGET_ORDER]

    for ex, x, y in zip(data, X, Y):
        pred = []
        for k in range(len(TARGET_ORDER)):
            pred.append(B[k] + dot(W[k], x))

        row = {
            "episode_index": ex.get("episode_index"),
            "selected_profile_name": ex["action"].get("selected_profile_name"),
            "target": {name: y[i] for i, name in enumerate(TARGET_ORDER)},
            "prediction": {name: pred[i] for i, name in enumerate(TARGET_ORDER)},
            "outcome": ex.get("outcome", {}),
            "obs": ex.get("obs", {}),
        }

        for k in range(len(TARGET_ORDER)):
            sq[k] += (pred[k] - y[k]) ** 2

        rows.append(row)

    rmse_by_target = {
        name: math.sqrt(sq[i] / max(n, 1))
        for i, name in enumerate(TARGET_ORDER)
    }

    model = {
        "model_type": "phase_b_theta_linear_regressor_v0",
        "feature_order": FEATURE_ORDER,
        "target_order": TARGET_ORDER,
        "feature_mean": {k: v for k, v in zip(FEATURE_ORDER, feat_mean)},
        "feature_std": {k: v for k, v in zip(FEATURE_ORDER, feat_std)},
        "bias": {k: v for k, v in zip(TARGET_ORDER, B)},
        "weights": {
            TARGET_ORDER[k]: {FEATURE_ORDER[j]: W[k][j] for j in range(len(FEATURE_ORDER))}
            for k in range(len(TARGET_ORDER))
        },
        "num_examples": n,
        "train_rmse_by_target": rmse_by_target,
        "train_predictions": rows,
        "output_limits": {
            "vx_far": [0.08, 0.16],
            "vx_near": [0.035, 0.08],
            "goal_slow_distance": [0.18, 0.30],
        },
        "notes": (
            "Continuous Phase-B theta/profile regressor trained from adaptive pseudo-RAM rollouts. "
            "This is a bootstrap supervised model before richer RAM/context-conditioned RL."
        ),
    }

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_model, "w") as f:
        json.dump(model, f, indent=2, sort_keys=True)

    print(json.dumps({
        "out_model": args.out_model,
        "num_examples": n,
        "train_rmse_by_target": rmse_by_target,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
