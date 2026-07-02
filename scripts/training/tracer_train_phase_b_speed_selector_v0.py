#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


FEATURE_ORDER = [
    "bias",
    "goal_distance_ahead",
    "vx_far",
    "vx_near",
    "goal_slow_distance",
    "goal_stop_distance",
    "vx_far_over_goal",
    "vx_near_over_goal",
    "slow_over_goal",
]


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def vec(features):
    return [float(features.get(k, 0.0)) for k in FEATURE_ORDER]


def dot(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))


def load_jsonl(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-jsonl", required=True)
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-4)
    args = ap.parse_args()

    data = load_jsonl(args.dataset_jsonl)
    if not data:
        raise RuntimeError(f"No preference data found: {args.dataset_jsonl}")

    w = [0.0 for _ in FEATURE_ORDER]

    for epoch in range(args.epochs):
        total_loss = 0.0

        for ex in data:
            xp = vec(ex["preferred"]["features"])
            xr = vec(ex["rejected"]["features"])
            dx = [a - b for a, b in zip(xp, xr)]

            z = dot(w, dx)
            p = sigmoid(z)
            loss = -math.log(max(p, 1e-9))
            total_loss += loss

            # d/dw -log(sigmoid(w·dx)) = (sigmoid(z)-1) dx
            grad_scale = p - 1.0
            for k in range(len(w)):
                grad = grad_scale * dx[k] + args.l2 * w[k]
                w[k] -= args.lr * grad

        if epoch in (0, args.epochs - 1) or (epoch + 1) % 200 == 0:
            avg_loss = total_loss / max(len(data), 1)
            print(f"[train] epoch={epoch+1} avg_loss={avg_loss:.6f}")

    correct = 0
    scored_examples = []
    for ex in data:
        xp = vec(ex["preferred"]["features"])
        xr = vec(ex["rejected"]["features"])

        sp = dot(w, xp)
        sr = dot(w, xr)
        ok = sp > sr
        correct += int(ok)

        scored_examples.append({
            "preferred_score_pred": sp,
            "rejected_score_pred": sr,
            "correct": ok,
            "margin_pred": sp - sr,
            "margin_teacher": ex.get("margin"),
            "preferred_summary": ex["preferred"]["summary_path"],
            "rejected_summary": ex["rejected"]["summary_path"],
        })

    model = {
        "model_type": "linear_pairwise_speed_selector_v0",
        "feature_order": FEATURE_ORDER,
        "weights": {k: v for k, v in zip(FEATURE_ORDER, w)},
        "num_examples": len(data),
        "train_pairwise_accuracy": correct / max(len(data), 1),
        "scored_examples": scored_examples,
        "notes": (
            "This is a small Phase-B teacher-style speed preference ranker. "
            "Use for candidate speed profile ranking under flat/low-mismatch conditions; "
            "replace or condition on RAM/context once richer rollout data is collected."
        ),
    }

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_model, "w") as f:
        json.dump(model, f, indent=2, sort_keys=True)

    print(json.dumps({
        "out_model": args.out_model,
        "num_examples": len(data),
        "train_pairwise_accuracy": model["train_pairwise_accuracy"],
        "weights": model["weights"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
