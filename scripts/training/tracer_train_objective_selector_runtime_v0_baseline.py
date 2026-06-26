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

DATA_CSV = ROOT / "data/preference_datasets/tracer_objective_selector_runtime_v0.csv"

OUT_DIR = ROOT / "artifacts/objective_selector_runtime_v0_baseline"
MODEL_JSON = OUT_DIR / "model.json"
EVAL_JSON = OUT_DIR / "eval_summary.json"
PRED_CSV = OUT_DIR / "predictions.csv"

TRACKED_MODEL_JSON = ROOT / "data/preference_datasets/tracer_objective_selector_runtime_v0_baseline_model.json"
TRACKED_EVAL_JSON = ROOT / "data/preference_datasets/tracer_objective_selector_runtime_v0_baseline_eval.json"
TRACKED_PRED_CSV = ROOT / "data/preference_datasets/tracer_objective_selector_runtime_v0_baseline_predictions.csv"

CATEGORICAL_FEATURES = [
    "terrain_family",
    "semantic_prior",
    "style_prior",
    "gate_first_action",
    "gate_last_action",
    "calibration_first",
    "calibration_last",
]

BOOL_FEATURES = [
    "ramgate_enabled",
    "ramgate_calibrated",
    "ramgate_observed",
    "ram_monitor_observed",
]

NUMERIC_FEATURES = [
    "ram_ctrl_risk",
    "ram_fallen_prob",
    "ram_recovery_prob",
    "gate_stable_prob",
    "gate_caution_prob",
    "gate_unstable_prob",
    "gate_override_prob",
]

LABEL = "semantic_target"
DEPLOY_LABEL = "deploy_label"
BETA_KEYS = ["beta_v_target", "beta_s_target", "beta_e_target"]


def f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def b(x: Any) -> float:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    return 1.0 if str(x).strip().lower() in {"1", "true", "yes", "y"} else 0.0


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def fit_vocab(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    vocab = {}
    for k in CATEGORICAL_FEATURES:
        vals = sorted({str(r.get(k, "")) for r in rows})
        vocab[k] = vals
    return vocab


def encode(row: dict[str, Any], vocab: dict[str, list[str]]) -> list[float]:
    x: list[float] = []

    for k in CATEGORICAL_FEATURES:
        val = str(row.get(k, ""))
        vals = vocab[k]
        x.extend([1.0 if val == v else 0.0 for v in vals])

    for k in BOOL_FEATURES:
        x.append(b(row.get(k, False)))

    for k in NUMERIC_FEATURES:
        x.append(f(row.get(k, 0.0), 0.0))

    return x


def feature_names(vocab: dict[str, list[str]]) -> list[str]:
    names = []
    for k in CATEGORICAL_FEATURES:
        for v in vocab[k]:
            names.append(f"{k}={v}")
    names.extend(BOOL_FEATURES)
    names.extend(NUMERIC_FEATURES)
    return names


def standardize_fit(xs: list[list[float]]) -> tuple[list[float], list[float]]:
    means = []
    stds = []
    for j in range(len(xs[0])):
        vals = [x[j] for x in xs]
        m = mean(vals)
        var = mean([(v - m) ** 2 for v in vals])
        s = math.sqrt(var)
        if s < 1e-8:
            s = 1.0
        means.append(m)
        stds.append(s)
    return means, stds


def standardize(x: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(v - m) / s for v, m, s in zip(x, means, stds)]


def dist2(a: list[float], c: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, c))


def fit_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vocab = fit_vocab(rows)
    xs_raw = [encode(r, vocab) for r in rows]
    means, stds = standardize_fit(xs_raw)
    xs = [standardize(x, means, stds) for x in xs_raw]

    by_label = defaultdict(list)
    for i, r in enumerate(rows):
        by_label[str(r[LABEL])].append(i)

    centroids = {}
    beta_by_label = {}
    deploy_by_label = {}
    class_counts = {}

    for label, idxs in by_label.items():
        centroids[label] = [
            mean([xs[i][j] for i in idxs])
            for j in range(len(xs[0]))
        ]
        beta = [
            mean([f(rows[i].get(k, 0.0), 0.0) for i in idxs])
            for k in BETA_KEYS
        ]
        s = sum(beta)
        if s > 1e-8:
            beta = [v / s for v in beta]
        beta_by_label[label] = beta

        deploy_counts = Counter(str(rows[i].get(DEPLOY_LABEL, "")) for i in idxs)
        deploy_by_label[label] = deploy_counts.most_common(1)[0][0]
        class_counts[label] = len(idxs)

    return {
        "kind": "runtime_nearest_centroid_objective_selector_v0",
        "categorical_features": CATEGORICAL_FEATURES,
        "bool_features": BOOL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "feature_names": feature_names(vocab),
        "vocab": vocab,
        "feature_mean": means,
        "feature_std": stds,
        "label": LABEL,
        "deploy_label": DEPLOY_LABEL,
        "beta_keys": BETA_KEYS,
        "centroids": centroids,
        "beta_by_label": beta_by_label,
        "deploy_by_label": deploy_by_label,
        "class_counts": class_counts,
        "training_source": str(DATA_CSV.relative_to(ROOT)),
    }


def predict(model: dict[str, Any], row: dict[str, Any]) -> tuple[str, str, list[float], dict[str, float]]:
    x = encode(row, model["vocab"])
    z = standardize(x, model["feature_mean"], model["feature_std"])

    scores = {}
    for label, c in model["centroids"].items():
        scores[label] = -dist2(z, c)

    pred = max(scores.items(), key=lambda kv: kv[1])[0]
    deploy = model["deploy_by_label"][pred]
    beta = model["beta_by_label"][pred]
    return pred, deploy, beta, scores


def evaluate(rows: list[dict[str, Any]], model: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preds = []
    sem_correct = 0
    deploy_correct = 0

    sem_conf = defaultdict(Counter)
    dep_conf = defaultdict(Counter)

    for r in rows:
        y = str(r[LABEL])
        yd = str(r[DEPLOY_LABEL])
        pred, dep, beta, scores = predict(model, r)

        if pred == y:
            sem_correct += 1
        if dep == yd:
            deploy_correct += 1

        sem_conf[y][pred] += 1
        dep_conf[yd][dep] += 1

        out = dict(r)
        out["pred_semantic_target"] = pred
        out["pred_deploy_label"] = dep
        out["pred_beta_v"] = beta[0]
        out["pred_beta_s"] = beta[1]
        out["pred_beta_e"] = beta[2]
        out["semantic_correct"] = pred == y
        out["deploy_correct"] = dep == yd
        out["score_json"] = json.dumps(scores)
        preds.append(out)

    labels = sorted(set([str(r[LABEL]) for r in rows]) | set(model["centroids"].keys()))
    deploy_labels = sorted(set([str(r[DEPLOY_LABEL]) for r in rows]))

    summary = {
        "num_samples": len(rows),
        "num_features": len(model["feature_names"]),
        "semantic_labels": labels,
        "deploy_labels": deploy_labels,
        "class_counts": dict(Counter(str(r[LABEL]) for r in rows)),
        "deploy_counts": dict(Counter(str(r[DEPLOY_LABEL]) for r in rows)),
        "semantic_accuracy_train_set": sem_correct / len(rows) if rows else 0.0,
        "deploy_accuracy_train_set": deploy_correct / len(rows) if rows else 0.0,
        "semantic_confusion": {k: dict(v) for k, v in sem_conf.items()},
        "deploy_confusion": {k: dict(v) for k, v in dep_conf.items()},
        "warning": (
            "Runtime v0 baseline uses runtime-feasible features only. "
            "It is still trained/evaluated on a small seed dataset and should be used "
            "for wiring/smoke tests before neural Objective Selector training."
        ),
    }

    return summary, preds


def main() -> None:
    rows = read_rows(DATA_CSV)
    if not rows:
        raise SystemExit(f"empty dataset: {DATA_CSV}")

    model = fit_model(rows)
    summary, preds = evaluate(rows, model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_JSON.write_text(json.dumps(model, indent=2) + "\n")
    EVAL_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    with PRED_CSV.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(preds[0].keys()))
        writer.writeheader()
        writer.writerows(preds)

    TRACKED_MODEL_JSON.write_text(json.dumps(model, indent=2) + "\n")
    TRACKED_EVAL_JSON.write_text(json.dumps(summary, indent=2) + "\n")
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
