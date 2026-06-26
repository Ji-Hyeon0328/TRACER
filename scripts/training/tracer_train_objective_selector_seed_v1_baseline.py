#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DATA_CSV = ROOT / "data/preference_datasets/tracer_objective_selector_seed_v1.csv"

OUT_DIR = ROOT / "artifacts/objective_selector_seed_v1_baseline"
MODEL_JSON = OUT_DIR / "model.json"
EVAL_JSON = OUT_DIR / "eval_summary.json"
PRED_CSV = OUT_DIR / "predictions.csv"

# Tracked copies. The artifacts/ directory is ignored by git, so keep
# compact reproducibility outputs under data/preference_datasets as well.
TRACKED_MODEL_JSON = ROOT / "data/preference_datasets/tracer_objective_selector_seed_v1_baseline_model.json"
TRACKED_EVAL_JSON = ROOT / "data/preference_datasets/tracer_objective_selector_seed_v1_baseline_eval.json"
TRACKED_PRED_CSV = ROOT / "data/preference_datasets/tracer_objective_selector_seed_v1_baseline_predictions.csv"

FEATURES = [
    "dx",
    "dy",
    "min_rel_z",
    "mpc_vx_mean",
    "mpc_body_height_mean",
    "mpc_clearance_mean",
    "ram_ctrl_risk_mean",
    "ram_fallen_mean",
    "ram_recovery_mean",
    "gate_stable_frac",
    "gate_caution_frac",
    "gate_unstable_frac",
    "gate_override_frac",
]

LABEL = "semantic_target"

BETA_KEYS = ["beta_v_target", "beta_s_target", "beta_e_target"]


def f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def row_features(row: dict[str, Any]) -> list[float]:
    return [f(row.get(k, 0.0), 0.0) for k in FEATURES]


def standardize_fit(xs: list[list[float]]) -> tuple[list[float], list[float]]:
    means = []
    stds = []
    for j in range(len(FEATURES)):
        vals = [x[j] for x in xs]
        m = mean(vals) if vals else 0.0
        var = mean([(v - m) ** 2 for v in vals]) if vals else 0.0
        s = math.sqrt(var)
        if s < 1e-8:
            s = 1.0
        means.append(m)
        stds.append(s)
    return means, stds


def standardize(x: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(v - m) / s for v, m, s in zip(x, means, stds)]


def dist2(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def centroid_fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    xs_raw = [row_features(r) for r in rows]
    means, stds = standardize_fit(xs_raw)
    xs = [standardize(x, means, stds) for x in xs_raw]

    by_label: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_label[str(r[LABEL])].append(i)

    centroids = {}
    beta_by_label = {}
    class_counts = {}

    for label, idxs in by_label.items():
        class_counts[label] = len(idxs)
        centroids[label] = [
            mean([xs[i][j] for i in idxs])
            for j in range(len(FEATURES))
        ]
        beta_by_label[label] = [
            mean([f(rows[i].get(k, 0.0), 0.0) for i in idxs])
            for k in BETA_KEYS
        ]

    return {
        "kind": "nearest_centroid_objective_selector_seed_v1",
        "features": FEATURES,
        "label": LABEL,
        "beta_keys": BETA_KEYS,
        "feature_mean": means,
        "feature_std": stds,
        "centroids": centroids,
        "beta_by_label": beta_by_label,
        "class_counts": class_counts,
    }


def predict(model: dict[str, Any], row: dict[str, Any]) -> tuple[str, list[float], dict[str, float]]:
    x = row_features(row)
    z = standardize(x, model["feature_mean"], model["feature_std"])

    scores = {}
    for label, c in model["centroids"].items():
        # lower distance is better
        scores[label] = -dist2(z, c)

    pred = max(scores.items(), key=lambda kv: kv[1])[0]
    beta = model["beta_by_label"][pred]
    s = sum(beta)
    if s > 1e-8:
        beta = [v / s for v in beta]

    return pred, beta, scores


def evaluate(rows: list[dict[str, Any]], model: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preds = []
    confusion = defaultdict(Counter)

    correct = 0
    for r in rows:
        y = str(r[LABEL])
        pred, beta, scores = predict(model, r)
        if pred == y:
            correct += 1
        confusion[y][pred] += 1

        out = dict(r)
        out["pred_semantic_target"] = pred
        out["pred_beta_v"] = beta[0]
        out["pred_beta_s"] = beta[1]
        out["pred_beta_e"] = beta[2]
        out["correct"] = pred == y
        out["score_json"] = json.dumps(scores)
        preds.append(out)

    labels = sorted(set([str(r[LABEL]) for r in rows]) | set(model["centroids"].keys()))
    per_label = {}
    for y in labels:
        total = sum(confusion[y].values())
        per_label[y] = {
            "total": total,
            "correct": confusion[y][y],
            "accuracy": (confusion[y][y] / total) if total else 0.0,
            "pred_counts": dict(confusion[y]),
        }

    eval_summary = {
        "num_samples": len(rows),
        "num_features": len(FEATURES),
        "labels": labels,
        "class_counts": dict(Counter(str(r[LABEL]) for r in rows)),
        "accuracy_train_set": correct / len(rows) if rows else 0.0,
        "per_label": per_label,
        "warning": (
            "This is a seed-data nearest-centroid baseline, not a final learned Objective Selector. "
            "Use it to validate dataset contract and runtime wiring before neural training."
        ),
    }

    return eval_summary, preds


def main() -> None:
    rows = read_rows(DATA_CSV)
    if not rows:
        raise SystemExit(f"empty dataset: {DATA_CSV}")

    model = centroid_fit(rows)
    eval_summary, preds = evaluate(rows, model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_JSON.write_text(json.dumps(model, indent=2) + "\n")
    EVAL_JSON.write_text(json.dumps(eval_summary, indent=2) + "\n")

    with PRED_CSV.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(preds[0].keys()))
        writer.writeheader()
        writer.writerows(preds)

    TRACKED_MODEL_JSON.parent.mkdir(parents=True, exist_ok=True)
    TRACKED_MODEL_JSON.write_text(json.dumps(model, indent=2) + "\n")
    TRACKED_EVAL_JSON.write_text(json.dumps(eval_summary, indent=2) + "\n")
    with TRACKED_PRED_CSV.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(preds[0].keys()))
        writer.writeheader()
        writer.writerows(preds)

    print("[TRACER] wrote", MODEL_JSON)
    print("[TRACER] wrote", EVAL_JSON)
    print("[TRACER] wrote", PRED_CSV)
    print("[TRACER] wrote", TRACKED_MODEL_JSON)
    print("[TRACER] wrote", TRACKED_EVAL_JSON)
    print("[TRACER] wrote", TRACKED_PRED_CSV)
    print(json.dumps(eval_summary, indent=2))


if __name__ == "__main__":
    main()
