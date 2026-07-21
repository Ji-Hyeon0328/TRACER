#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

TARGETS = [
    "target_beta_motion_true_metric",
    "target_beta_stability_true_metric",
    "target_beta_energy_true_metric",
]

NUMERIC_CANDIDATES = [
    "x", "y", "reset_y", "f7_reset_y", "y_mean",
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
    "true_metric_base_vx_mean",
    "true_metric_base_vy_abs_mean",
    "true_metric_joint_abs_power_mean",
    "true_metric_imu_ang_vel_norm_mean",
    "true_metric_contact_force_z_sum_mean",
]

def ffloat(x):
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None

def beta_normalize(y):
    y = np.maximum(y, 1e-6)
    z = y.sum(axis=1, keepdims=True)
    return y / np.maximum(z, 1e-12)

def build_design(rows, feature_spec=None, fit=True):
    if fit:
        contexts = sorted(set(r.get("context", "unknown") or "unknown" for r in rows))
        numeric_features = []
        for k in NUMERIC_CANDIDATES:
            vals = [ffloat(r.get(k)) for r in rows]
            vals = [v for v in vals if v is not None]
            if vals:
                numeric_features.append(k)
        feature_spec = {
            "contexts": contexts,
            "numeric_features": numeric_features,
        }

    contexts = feature_spec["contexts"]
    numeric_features = feature_spec["numeric_features"]

    X_parts = []

    # context one-hot
    ctx_mat = np.zeros((len(rows), len(contexts)), dtype=float)
    ctx_to_i = {c: i for i, c in enumerate(contexts)}
    for i, r in enumerate(rows):
        c = r.get("context", "unknown") or "unknown"
        if c in ctx_to_i:
            ctx_mat[i, ctx_to_i[c]] = 1.0
    X_parts.append(ctx_mat)

    # numeric
    num_mat = np.zeros((len(rows), len(numeric_features)), dtype=float)
    for j, k in enumerate(numeric_features):
        for i, r in enumerate(rows):
            v = ffloat(r.get(k))
            num_mat[i, j] = 0.0 if v is None else v

    if fit:
        mu = num_mat.mean(axis=0) if num_mat.shape[1] else np.zeros((0,), dtype=float)
        sd = num_mat.std(axis=0) if num_mat.shape[1] else np.ones((0,), dtype=float)
        sd = np.where(sd < 1e-9, 1.0, sd)
        feature_spec["numeric_mean"] = mu.tolist()
        feature_spec["numeric_std"] = sd.tolist()
    else:
        mu = np.array(feature_spec["numeric_mean"], dtype=float)
        sd = np.array(feature_spec["numeric_std"], dtype=float)

    if num_mat.shape[1]:
        num_mat = (num_mat - mu) / sd
        X_parts.append(num_mat)

    X = np.concatenate(X_parts, axis=1) if X_parts else np.zeros((len(rows), 0), dtype=float)
    X = np.concatenate([np.ones((len(rows), 1), dtype=float), X], axis=1)

    feature_names = (
        ["bias"] +
        [f"context={c}" for c in contexts] +
        numeric_features
    )

    return X, feature_spec, feature_names

def read_rows(p):
    return list(csv.DictReader(open(p)))

def targets(rows):
    Y = []
    for r in rows:
        vals = [ffloat(r.get(k)) for k in TARGETS]
        if any(v is None for v in vals):
            raise RuntimeError("missing target beta in row")
        Y.append(vals)
    return np.array(Y, dtype=float)

def fit_ridge(X, Y, alpha):
    A = X.T @ X + alpha * np.eye(X.shape[1])
    A[0, 0] -= alpha  # do not regularize bias
    B = X.T @ Y
    return np.linalg.solve(A, B)

def predict_beta(X, W):
    return beta_normalize(X @ W)

def rmse_mae(Y, P):
    err = P - Y
    rmse = np.sqrt(np.mean(err * err, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    return rmse, mae

def mean_beta(rows, P):
    by_tag = {}
    for tag in sorted(set(r.get("f7_source_tag", "unknown") for r in rows)):
        idx = [i for i, r in enumerate(rows) if r.get("f7_source_tag", "unknown") == tag]
        by_tag[tag] = P[idx].mean(axis=0).tolist()
    return by_tag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="datasets/phase_f/f7_d7_true_metric_calibration_dataset_v0.csv")
    ap.add_argument("--model-json", default="models/phase_f/f8_d7_true_metric_beta_calibrator_ridge_v0.json")
    ap.add_argument("--out-md", default="reports/phase_f/f8_d7_true_metric_beta_calibrator_summary_v0.md")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    rows = read_rows(args.in_csv)
    if not rows:
        raise SystemExit("empty input")

    Y = targets(rows)
    X, spec, feature_names = build_design(rows, fit=True)
    W = fit_ridge(X, Y, args.alpha)
    P = predict_beta(X, W)

    rmse, mae = rmse_mae(Y, P)

    tags = sorted(set(r.get("f7_source_tag", "unknown") for r in rows))
    tag_summary = []
    for tag in tags:
        idx = [i for i, r in enumerate(rows) if r.get("f7_source_tag", "unknown") == tag]
        yt = Y[idx].mean(axis=0)
        pt = P[idx].mean(axis=0)
        tag_summary.append({
            "tag": tag,
            "rows": len(idx),
            "target": yt.tolist(),
            "pred": pt.tolist(),
            "abs_err": np.abs(pt - yt).tolist(),
        })

    # Leave-one-source-tag-out diagnostic
    loco = []
    for hold_tag in tags:
        train_rows = [r for r in rows if r.get("f7_source_tag", "unknown") != hold_tag]
        test_rows = [r for r in rows if r.get("f7_source_tag", "unknown") == hold_tag]
        if not train_rows or not test_rows:
            continue

        Ytr = targets(train_rows)
        Yte = targets(test_rows)
        Xtr, sp2, _ = build_design(train_rows, fit=True)
        Xte, _, _ = build_design(test_rows, feature_spec=sp2, fit=False)
        W2 = fit_ridge(Xtr, Ytr, args.alpha)
        Pte = predict_beta(Xte, W2)
        r2, m2 = rmse_mae(Yte, Pte)

        loco.append({
            "heldout_tag": hold_tag,
            "rows": len(test_rows),
            "target_mean": Yte.mean(axis=0).tolist(),
            "pred_mean": Pte.mean(axis=0).tolist(),
            "rmse": r2.tolist(),
            "mae": m2.tolist(),
        })

    model = {
        "version": "phase_f8_d7_true_metric_beta_calibrator_ridge_v0",
        "input_csv": args.in_csv,
        "alpha": args.alpha,
        "target_names": TARGETS,
        "feature_names": feature_names,
        "feature_spec": spec,
        "weights": W.tolist(),
        "train_rmse": rmse.tolist(),
        "train_mae": mae.tolist(),
        "tag_summary": tag_summary,
        "leave_one_tag_out": loco,
        "safe_interpretation": (
            "Diagnostic calibration model only. Data is timestep-expanded from three rollout conditions, "
            "so this should not replace runtime D7 without more rollouts."
        ),
    }

    model_path = Path(args.model_json)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2))

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("# TRACER Phase-F8 D7 True-Metric Beta Calibrator v0\n\n")
        f.write("This trains a ridge calibration model from D7 runtime features to Phase-F6 true-metric beta target seeds.\n\n")
        f.write(f"- input csv: `{args.in_csv}`\n")
        f.write(f"- model json: `{model_path}`\n")
        f.write(f"- rows: `{len(rows)}`\n")
        f.write(f"- features: `{len(feature_names)}`\n")
        f.write(f"- alpha: `{args.alpha}`\n\n")

        f.write("## Train error\n\n")
        f.write("| beta | RMSE | MAE |\n")
        f.write("|---|---:|---:|\n")
        for name, r, m in zip(["motion", "stability", "energy"], rmse, mae):
            f.write(f"| {name} | {r:.6f} | {m:.6f} |\n")

        f.write("\n## Per-condition mean prediction\n\n")
        f.write("| tag | rows | target beta | predicted beta | abs error |\n")
        f.write("|---|---:|---|---|---|\n")
        for s in tag_summary:
            tb = ", ".join(f"{x:.4f}" for x in s["target"])
            pb = ", ".join(f"{x:.4f}" for x in s["pred"])
            eb = ", ".join(f"{x:.4f}" for x in s["abs_err"])
            f.write(f"| {s['tag']} | {s['rows']} | ({tb}) | ({pb}) | ({eb}) |\n")

        f.write("\n## Leave-one-tag-out diagnostic\n\n")
        f.write("| heldout | rows | target mean | predicted mean | RMSE |\n")
        f.write("|---|---:|---|---|---|\n")
        for s in loco:
            tb = ", ".join(f"{x:.4f}" for x in s["target_mean"])
            pb = ", ".join(f"{x:.4f}" for x in s["pred_mean"])
            rb = ", ".join(f"{x:.4f}" for x in s["rmse"])
            f.write(f"| {s['heldout_tag']} | {s['rows']} | ({tb}) | ({pb}) | ({rb}) |\n")

        f.write("\n## Safe interpretation\n\n")
        f.write("This is a diagnostic calibration model, not a deployable replacement for D7. The dataset has many timestep rows but only three source rollout conditions, so train error can be optimistic. Use this to quantify the gap between existing D7 runtime beta behavior and true-metric beta seeds.\n")

    print(f"[TRACER] wrote {model_path}")
    print(f"[TRACER] wrote {out_md}")
    print(f"[TRACER] rows={len(rows)} features={len(feature_names)}")
    print("[TRACER] train RMSE:", ", ".join(f"{x:.6f}" for x in rmse))
    for s in tag_summary:
        print(
            f"[TRACER] {s['tag']}: target={tuple(round(x,4) for x in s['target'])} "
            f"pred={tuple(round(x,4) for x in s['pred'])}"
        )

if __name__ == "__main__":
    main()
