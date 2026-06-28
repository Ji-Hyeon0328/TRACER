#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


STYLES = ["fast", "cautious", "high_clearance"]
OBJECTIVES = ["motion_objective", "stability_objective", "deploy_objective"]
TERRAINS = ["flat_normal", "rough_mid", "slope_5deg"]


def f(x, default=0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def one_hot(name: str, choices: List[str]) -> List[float]:
    return [1.0 if name == c else 0.0 for c in choices]


def candidate_metrics(row: dict, prefix: str) -> dict:
    success = f(row.get(f"{prefix}_success_rate"))
    distance = f(row.get(f"{prefix}_distance_mean"))
    progress = f(row.get(f"{prefix}_progress_mean"))
    deploy = f(row.get(f"{prefix}_deploy_mean"))
    fallen = f(row.get(f"{prefix}_fallen_p90_max"))
    fresh = f(row.get(f"{prefix}_fresh1_min"))

    # Simple proxies.
    stability_score = 0.45 * success + 0.25 * fresh + 0.30 * max(0.0, 1.0 - fallen)
    motion_score = 0.65 * distance + 0.35 * progress
    deploy_score = deploy

    # Energy proxy: lower aggressive motion is treated as more energy-conservative.
    # This is crude, but useful as a v0 feature until torque/CoT is logged.
    energy_score = max(0.0, 1.0 - min(1.0, distance))

    return {
        "success_rate": success,
        "distance_mean": distance,
        "progress_mean": progress,
        "deploy_mean": deploy,
        "fallen_p90_max": fallen,
        "fresh1_min": fresh,
        "motion_score_proxy": motion_score,
        "stability_score_proxy": stability_score,
        "deploy_score_proxy": deploy_score,
        "energy_score_proxy": energy_score,
    }


def build_feature_names() -> List[str]:
    names: List[str] = []

    names += [f"terrain={t}" for t in TERRAINS]
    names += [f"objective={o}" for o in OBJECTIVES]
    names += ["beta_motion", "beta_stability", "beta_energy"]

    names += [f"style={s}" for s in STYLES]

    # β × style interaction. This is important because terrain/beta alone cancel
    # inside pairwise differences, but style-conditioned β preference does not.
    for s in STYLES:
        names += [f"style={s}:beta_motion", f"style={s}:beta_stability", f"style={s}:beta_energy"]

    metric_names = [
        "success_rate",
        "distance_mean",
        "progress_mean",
        "deploy_mean",
        "fallen_p90_max",
        "fresh1_min",
        "motion_score_proxy",
        "stability_score_proxy",
        "deploy_score_proxy",
        "energy_score_proxy",
    ]
    names += metric_names

    # β × metric interactions encode the IRL-like reward decomposition:
    # R_beta(tau) ≈ beta_motion * motion_features
    #             + beta_stability * stability_features
    #             + beta_energy * energy_features
    for metric in metric_names:
        names += [
            f"beta_motion:{metric}",
            f"beta_stability:{metric}",
            f"beta_energy:{metric}",
        ]

    return names


FEATURE_NAMES = build_feature_names()


def candidate_features(row: dict, style: str, prefix: str) -> np.ndarray:
    terrain = row.get("terrain", "unknown")
    objective = row.get("objective_name", "unknown")
    beta_m = f(row.get("beta_motion"))
    beta_s = f(row.get("beta_stability"))
    beta_e = f(row.get("beta_energy"))
    metrics = candidate_metrics(row, prefix)

    feats: List[float] = []
    feats += one_hot(terrain, TERRAINS)
    feats += one_hot(objective, OBJECTIVES)
    feats += [beta_m, beta_s, beta_e]

    style_oh = one_hot(style, STYLES)
    feats += style_oh

    for s_idx, _s in enumerate(STYLES):
        active = style_oh[s_idx]
        feats += [active * beta_m, active * beta_s, active * beta_e]

    metric_order = [
        "success_rate",
        "distance_mean",
        "progress_mean",
        "deploy_mean",
        "fallen_p90_max",
        "fresh1_min",
        "motion_score_proxy",
        "stability_score_proxy",
        "deploy_score_proxy",
        "energy_score_proxy",
    ]

    metric_vals = [metrics[k] for k in metric_order]
    feats += metric_vals

    for v in metric_vals:
        feats += [beta_m * v, beta_s * v, beta_e * v]

    arr = np.asarray(feats, dtype=np.float64)
    if arr.shape[0] != len(FEATURE_NAMES):
        raise RuntimeError(f"feature length mismatch: {arr.shape[0]} vs {len(FEATURE_NAMES)}")
    return arr


def load_pairs(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    rows = []
    xdiffs = []
    labels = []
    weights = []

    with path.open(newline="") as fp:
        for row in csv.DictReader(fp):
            winner_style = row["winner_style"]
            loser_style = row["loser_style"]

            xw = candidate_features(row, winner_style, "winner")
            xl = candidate_features(row, loser_style, "loser")
            xd = xw - xl

            margin = f(row.get("margin"), 0.0)
            conf = f(row.get("label_confidence"), 1.0)

            # Keep all rows, but weak labels get smaller weight.
            w = max(0.05, conf) * max(0.02, min(1.0, margin))

            xdiffs.append(xd)
            labels.append(1.0)
            weights.append(w)
            rows.append(row)

    return (
        np.vstack(xdiffs).astype(np.float64),
        np.asarray(labels, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        rows,
    )


def train_logistic_pairwise(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    seed: int = 7,
    epochs: int = 3000,
    lr: float = 0.05,
    l2: float = 1e-3,
) -> dict:
    rng = np.random.default_rng(seed)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    xs = (x - mean) / std

    n, d = xs.shape
    w = rng.normal(0.0, 0.01, size=d)
    b = 0.0

    sw = sample_weight / (sample_weight.mean() + 1e-12)

    history = []
    for ep in range(epochs):
        logits = xs @ w + b
        p = sigmoid(logits)

        # y is always 1.0 here because x is winner-loser.
        err = (p - y) * sw

        grad_w = (xs.T @ err) / n + l2 * w
        grad_b = float(err.mean())

        w -= lr * grad_w
        b -= lr * grad_b

        if ep % 250 == 0 or ep == epochs - 1:
            eps = 1e-9
            loss = -np.mean(sw * (y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
            loss += 0.5 * l2 * float(np.sum(w * w))
            acc = float(np.mean((p >= 0.5) == (y >= 0.5)))
            history.append({"epoch": ep, "loss": float(loss), "acc": acc})

    logits = xs @ w + b
    probs = sigmoid(logits)
    acc = float(np.mean(probs >= 0.5))

    return {
        "weights": w,
        "bias": b,
        "mean": mean,
        "std": std,
        "history": history,
        "train_probs": probs,
        "train_acc": acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pairs-csv",
        default="data/preference_datasets/tracer_objective_conditioned_style_pairs_3x3_v0.csv",
    )
    ap.add_argument(
        "--artifact-json",
        default="artifacts/tracer_preference_objective_irl_v0/model.json",
    )
    ap.add_argument(
        "--report-csv",
        default="reports/tracer_preference_objective_irl_v0_train_report.csv",
    )
    ap.add_argument(
        "--summary-json",
        default="reports/tracer_preference_objective_irl_v0_summary.json",
    )
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    pairs_path = Path(args.pairs_csv)
    x, y, sw, rows = load_pairs(pairs_path)

    result = train_logistic_pairwise(
        x,
        y,
        sw,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
    )

    out_model = Path(args.artifact_json)
    out_report = Path(args.report_csv)
    out_summary = Path(args.summary_json)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    w = result["weights"]
    b = result["bias"]
    mean = result["mean"]
    std = result["std"]
    probs = result["train_probs"]

    model = {
        "version": "tracer_preference_objective_irl_v0",
        "source_pairs_csv": str(pairs_path),
        "model_type": "linear_pairwise_bradley_terry",
        "feature_names": FEATURE_NAMES,
        "weights": w.tolist(),
        "bias": float(b),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "train_acc": float(result["train_acc"]),
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "l2": args.l2,
            "seed": args.seed,
            "history": result["history"],
        },
        "interpretation": (
            "The model scores candidate rollout/style outcomes under terrain and beta. "
            "Pairwise preference probability is sigmoid(score(winner)-score(loser)). "
            "This is a v0 preference-IRL reward teacher for TRACER Objective Selector training."
        ),
    }
    out_model.write_text(json.dumps(model, indent=2))

    fields = [
        "terrain",
        "objective_name",
        "beta_motion",
        "beta_stability",
        "beta_energy",
        "winner_style",
        "loser_style",
        "margin",
        "label_confidence",
        "sample_weight",
        "pred_pref_prob",
        "pred_correct",
    ]

    with out_report.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row, weight, prob in zip(rows, sw, probs):
            writer.writerow(
                {
                    "terrain": row.get("terrain"),
                    "objective_name": row.get("objective_name"),
                    "beta_motion": row.get("beta_motion"),
                    "beta_stability": row.get("beta_stability"),
                    "beta_energy": row.get("beta_energy"),
                    "winner_style": row.get("winner_style"),
                    "loser_style": row.get("loser_style"),
                    "margin": row.get("margin"),
                    "label_confidence": row.get("label_confidence"),
                    "sample_weight": weight,
                    "pred_pref_prob": float(prob),
                    "pred_correct": int(prob >= 0.5),
                }
            )

    # Top coefficients for interpretability.
    coef_rows = sorted(
        [
            {
                "feature": name,
                "weight": float(weight),
                "abs_weight": abs(float(weight)),
            }
            for name, weight in zip(FEATURE_NAMES, w)
        ],
        key=lambda r: r["abs_weight"],
        reverse=True,
    )

    summary = {
        "n_pairs": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "train_acc": float(result["train_acc"]),
        "artifact_json": str(out_model),
        "report_csv": str(out_report),
        "top_positive_features": [r for r in coef_rows if r["weight"] > 0][:15],
        "top_negative_features": [r for r in coef_rows if r["weight"] < 0][:15],
        "history_tail": result["history"][-5:],
    }
    out_summary.write_text(json.dumps(summary, indent=2))

    print(f"[TRACER] pairs={x.shape[0]} features={x.shape[1]} train_acc={result['train_acc']:.3f}")
    print(f"[TRACER] wrote {out_model}")
    print(f"[TRACER] wrote {out_report}")
    print(f"[TRACER] wrote {out_summary}")
    print()
    print("[top positive]")
    for r in summary["top_positive_features"][:10]:
        print(f"  {r['feature']:45s} {r['weight']:+.4f}")
    print("[top negative]")
    for r in summary["top_negative_features"][:10]:
        print(f"  {r['feature']:45s} {r['weight']:+.4f}")


if __name__ == "__main__":
    main()
