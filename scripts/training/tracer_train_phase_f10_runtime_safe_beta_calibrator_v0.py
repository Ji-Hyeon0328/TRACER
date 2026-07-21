#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

IN_CSV = Path("datasets/phase_f/f7_d7_true_metric_calibration_dataset_v0.csv")
OUT_MODEL = Path("models/phase_f/f10_runtime_safe_beta_calibrator_ridge_v0.json")
OUT_MD = Path("reports/phase_f/f10_runtime_safe_beta_calibrator_summary_v0.md")

ALPHA = 1.0

TARGETS = [
    "target_beta_motion_true_metric",
    "target_beta_stability_true_metric",
    "target_beta_energy_true_metric",
]

# Runtime-safe only.
# Exclude:
#   f7_source_tag
#   f7_reset_y
#   true_metric_*
#   target_*
#   summary/source labels
NUMERIC_FEATURES = [
    "x", "y", "y_mean",
    "pred_beta_motion", "pred_beta_stability", "pred_beta_energy",
    "raw_beta_motion", "raw_beta_stability", "raw_beta_energy",
    "prior_beta_motion", "prior_beta_stability", "prior_beta_energy",
    "actual_beta_motion", "actual_beta_stability", "actual_beta_energy",
    "motion_score", "stability_score", "energy_score",
    "effort_proxy", "hold_drift",
    "rollout_max_abs_y", "rollout_mean_abs_y",
    "mean_abs_y_context", "max_abs_y_context",
    "ram_slip_proxy_mean", "ram_roughness_proxy_mean", "ram_sigma_mean",
    "ref_vx_mean", "ref_yaw_rate_mean", "ref_clearance_mean",
]

def ff(x, default=None):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default

def read_rows(p):
    return list(csv.DictReader(open(p)))

def beta_normalize(Y):
    Y = np.maximum(Y, 1e-9)
    return Y / np.maximum(Y.sum(axis=1, keepdims=True), 1e-12)

def build_X(rows, spec=None, fit=True):
    if fit:
        contexts = sorted(set((r.get("context") or "unknown") for r in rows))
        numeric = []
        for k in NUMERIC_FEATURES:
            vals = [ff(r.get(k)) for r in rows]
            vals = [v for v in vals if v is not None]
            if vals:
                numeric.append(k)
        spec = {"contexts": contexts, "numeric_features": numeric}

    contexts = spec["contexts"]
    numeric = spec["numeric_features"]

    X_parts = []

    C = np.zeros((len(rows), len(contexts)))
    cidx = {c: i for i, c in enumerate(contexts)}
    for i, r in enumerate(rows):
        c = r.get("context") or "unknown"
        if c in cidx:
            C[i, cidx[c]] = 1.0
    X_parts.append(C)

    N = np.zeros((len(rows), len(numeric)))
    for j, k in enumerate(numeric):
        for i, r in enumerate(rows):
            N[i, j] = ff(r.get(k), 0.0)

    if fit:
        mu = N.mean(axis=0) if N.shape[1] else np.zeros((0,))
        sd = N.std(axis=0) if N.shape[1] else np.ones((0,))
        sd = np.where(sd < 1e-9, 1.0, sd)
        spec["numeric_mean"] = mu.tolist()
        spec["numeric_std"] = sd.tolist()
    else:
        mu = np.array(spec["numeric_mean"])
        sd = np.array(spec["numeric_std"])

    if N.shape[1]:
        N = (N - mu) / sd
        X_parts.append(N)

    X = np.concatenate(X_parts, axis=1)
    X = np.concatenate([np.ones((len(rows), 1)), X], axis=1)

    feature_names = ["bias"] + [f"context={c}" for c in contexts] + numeric
    return X, spec, feature_names

def build_Y(rows):
    Y = []
    for r in rows:
        vals = [ff(r.get(k)) for k in TARGETS]
        if any(v is None for v in vals):
            raise RuntimeError("missing target beta")
        Y.append(vals)
    return np.array(Y)

def fit_ridge(X, Y):
    A = X.T @ X + ALPHA * np.eye(X.shape[1])
    A[0, 0] -= ALPHA
    B = X.T @ Y
    return np.linalg.solve(A, B)

def predict(X, W):
    return beta_normalize(X @ W)

def rmse_mae(Y, P):
    E = P - Y
    return np.sqrt(np.mean(E * E, axis=0)), np.mean(np.abs(E), axis=0)

def vec_mean(V):
    if len(V) == 0:
        return np.array([np.nan, np.nan, np.nan])
    return np.mean(np.array(V), axis=0)

def fmtv(v):
    return "(" + ", ".join(f"{float(x):.4f}" for x in v) + ")"

rows = read_rows(IN_CSV)
Y = build_Y(rows)
X, spec, feature_names = build_X(rows, fit=True)
W = fit_ridge(X, Y)
P = predict(X, W)
rmse, mae = rmse_mae(Y, P)

tags = sorted(set(r.get("f7_source_tag", "unknown") for r in rows))

tag_summary = []
for tag in tags:
    idx = [i for i, r in enumerate(rows) if r.get("f7_source_tag") == tag]
    yt = Y[idx].mean(axis=0)
    pt = P[idx].mean(axis=0)
    tag_summary.append({
        "tag": tag,
        "rows": len(idx),
        "target": yt.tolist(),
        "pred": pt.tolist(),
        "abs_err": np.abs(pt - yt).tolist(),
        "l1": float(np.abs(pt - yt).sum()),
    })

loco = []
for hold in tags:
    train_rows = [r for r in rows if r.get("f7_source_tag") != hold]
    test_rows = [r for r in rows if r.get("f7_source_tag") == hold]
    Ytr = build_Y(train_rows)
    Yte = build_Y(test_rows)
    Xtr, sp, _ = build_X(train_rows, fit=True)
    Xte, _, _ = build_X(test_rows, spec=sp, fit=False)
    W2 = fit_ridge(Xtr, Ytr)
    Pte = predict(Xte, W2)
    r, m = rmse_mae(Yte, Pte)
    loco.append({
        "heldout": hold,
        "rows": len(test_rows),
        "target": Yte.mean(axis=0).tolist(),
        "pred": Pte.mean(axis=0).tolist(),
        "rmse": r.tolist(),
        "mae": m.tolist(),
        "l1_mean": float(np.mean(np.sum(np.abs(Pte - Yte), axis=1))),
    })

OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
OUT_MODEL.write_text(json.dumps({
    "version": "phase_f10_runtime_safe_beta_calibrator_ridge_v0",
    "input_csv": str(IN_CSV),
    "alpha": ALPHA,
    "target_names": TARGETS,
    "feature_names": feature_names,
    "feature_spec": spec,
    "weights": W.tolist(),
    "train_rmse": rmse.tolist(),
    "train_mae": mae.tolist(),
    "tag_summary": tag_summary,
    "leave_one_tag_out": loco,
    "excluded_features_note": "Excluded f7_source_tag, f7_reset_y, true_metric_*, and target_* from runtime-safe features.",
    "safe_interpretation": "Diagnostic runtime-safe model only. Not deployable until trained on more rollout conditions.",
}, indent=2))

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F10 Runtime-Safe Beta Calibrator v0\n\n")
    f.write("This trains a diagnostic beta calibrator using only runtime-available D7/log features, excluding true-metric target leakage.\n\n")
    f.write(f"- input: `{IN_CSV}`\n")
    f.write(f"- model: `{OUT_MODEL}`\n")
    f.write(f"- rows: `{len(rows)}`\n")
    f.write(f"- features: `{len(feature_names)}`\n")
    f.write(f"- alpha: `{ALPHA}`\n\n")

    f.write("## Train error\n\n")
    f.write("| beta | RMSE | MAE |\n")
    f.write("|---|---:|---:|\n")
    for name, r, m in zip(["motion", "stability", "energy"], rmse, mae):
        f.write(f"| {name} | {r:.6f} | {m:.6f} |\n")

    f.write("\n## Per-condition train-fit mean\n\n")
    f.write("| tag | rows | target beta | predicted beta | L1 mean error |\n")
    f.write("|---|---:|---|---|---:|\n")
    for s in tag_summary:
        f.write(f"| {s['tag']} | {s['rows']} | {fmtv(s['target'])} | {fmtv(s['pred'])} | {s['l1']:.6f} |\n")

    f.write("\n## Leave-one-tag-out diagnostic\n\n")
    f.write("| heldout | rows | target mean | predicted mean | mean L1 error | RMSE |\n")
    f.write("|---|---:|---|---|---:|---|\n")
    for s in loco:
        f.write(
            f"| {s['heldout']} | {s['rows']} | {fmtv(s['target'])} | "
            f"{fmtv(s['pred'])} | {s['l1_mean']:.6f} | {fmtv(s['rmse'])} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("This is stricter than F8 because it removes true-metric leakage features. If leave-one-tag-out is weak, that means the current three-condition dataset is insufficient for a deployable beta calibrator. Use this as a diagnostic before expanding F3 data.\n")

print(f"[TRACER] wrote {OUT_MODEL}")
print(f"[TRACER] wrote {OUT_MD}")
print(f"[TRACER] train RMSE={fmtv(rmse)}")
for s in tag_summary:
    print(f"[TRACER] {s['tag']}: target={fmtv(s['target'])} pred={fmtv(s['pred'])} L1={s['l1']:.6f}")
print("[TRACER] leave-one-tag-out:")
for s in loco:
    print(f"  {s['heldout']}: pred={fmtv(s['pred'])} target={fmtv(s['target'])} L1={s['l1_mean']:.6f}")
