#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


STYLES = ["fast", "cautious", "high_clearance"]
MISSIONS = ["motion", "deploy", "stability"]
RISK_LEVELS = ["low", "mid", "high"]


def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def one_hot(v, choices):
    return [1.0 if v == c else 0.0 for c in choices]


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def build_features(rows):
    terrains = sorted({r["terrain"] for r in rows})

    feature_names = []
    for t in terrains:
        feature_names.append(f"terrain={t}")
    for m in MISSIONS:
        feature_names.append(f"mission={m}")
    for r in RISK_LEVELS:
        feature_names.append(f"risk={r}")
    feature_names += [
        "ram_risk",
        "recovery_needed",
        "beta_motion",
        "beta_stability",
        "beta_energy",
        "ram_risk:beta_stability",
        "ram_risk:beta_motion",
        "recovery_needed:beta_stability",
    ]

    xs = []
    ys = []
    for r in rows:
        bm = f(r["beta_motion"])
        bs = f(r["beta_stability"])
        be = f(r["beta_energy"])
        rr = f(r["ram_risk"])
        rec = f(r["recovery_needed"])

        x = []
        x += one_hot(r["terrain"], terrains)
        x += one_hot(r["mission"], MISSIONS)
        x += one_hot(r["risk_level"], RISK_LEVELS)
        x += [rr, rec, bm, bs, be, rr * bs, rr * bm, rec * bs]

        xs.append(x)
        ys.append(STYLES.index(r["target_style"]))

    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64), feature_names, terrains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-csv", default="data/gms/tracer_gms_v1_risk_aware_targets.csv")
    ap.add_argument("--artifact-json", default="artifacts/tracer_gms_supervised_v1/model.json")
    ap.add_argument("--report-csv", default="reports/tracer_gms_supervised_v1_train_report.csv")
    ap.add_argument("--summary-json", default="reports/tracer_gms_supervised_v1_summary.json")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--l2", type=float, default=1e-4)
    args = ap.parse_args()

    with Path(args.targets_csv).open(newline="") as fp:
        rows = list(csv.DictReader(fp))

    X, y, feature_names, terrains = build_features(rows)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xs = (X - mean) / std

    n, d = Xs.shape
    k = len(STYLES)

    rng = np.random.default_rng(7)
    W = rng.normal(0.0, 0.01, size=(d, k))
    b = np.zeros(k, dtype=np.float64)

    Y = np.zeros((n, k), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    for epoch in range(args.epochs):
        logits = Xs @ W + b
        P = softmax(logits)
        grad_logits = (P - Y) / n

        grad_W = Xs.T @ grad_logits + args.l2 * W
        grad_b = grad_logits.sum(axis=0)

        W -= args.lr * grad_W
        b -= args.lr * grad_b

    P = softmax(Xs @ W + b)
    pred = np.argmax(P, axis=1)
    acc = float(np.mean(pred == y))

    out_artifact = Path(args.artifact_json)
    out_report = Path(args.report_csv)
    out_summary = Path(args.summary_json)

    out_artifact.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_type": "softmax_linear_gms_supervised_v1",
        "styles": STYLES,
        "missions": MISSIONS,
        "risk_levels": RISK_LEVELS,
        "terrains": terrains,
        "feature_names": feature_names,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "weights": W.tolist(),
        "bias": b.tolist(),
        "train_acc": acc,
        "targets_csv": args.targets_csv,
    }
    out_artifact.write_text(json.dumps(artifact, indent=2))

    with out_report.open("w", newline="") as fp:
        fields = [
            "terrain", "mission", "risk_level",
            "target_style", "pred_style", "pred_conf",
            "p_fast", "p_cautious", "p_high_clearance",
        ]
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r, pi, yi, yh in zip(rows, P, y, pred):
            w.writerow({
                "terrain": r["terrain"],
                "mission": r["mission"],
                "risk_level": r["risk_level"],
                "target_style": STYLES[yi],
                "pred_style": STYLES[yh],
                "pred_conf": float(pi[yh]),
                "p_fast": float(pi[0]),
                "p_cautious": float(pi[1]),
                "p_high_clearance": float(pi[2]),
            })

    summary = {
        "targets_csv": args.targets_csv,
        "artifact_json": str(out_artifact),
        "report_csv": str(out_report),
        "rows": int(n),
        "features": int(d),
        "train_acc": acc,
    }
    out_summary.write_text(json.dumps(summary, indent=2))

    print("[TRACER] GMS supervised v1")
    print(json.dumps(summary, indent=2))

    print("\n[TRACER] predictions")
    for r, pi, yi, yh in zip(rows, P, y, pred):
        print(
            f"{r['terrain']:38s} "
            f"mission={r['mission']:9s} "
            f"risk={r['risk_level']:4s} "
            f"target={STYLES[yi]:14s} "
            f"pred={STYLES[yh]:14s} "
            f"conf={float(pi[yh]):.3f}"
        )


if __name__ == "__main__":
    main()
