#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np

IN_CSV = Path("datasets/phase_i/i0_ram_teacher_dataset_v0.csv")
IN_MANIFEST = Path("datasets/phase_i/i0_ram_teacher_manifest_v0.json")

OUT_MODEL = Path("models/phase_i/i1_ram_ridge_predictor_v0.json")
OUT_EVAL = Path("datasets/phase_i/i1_ram_ridge_loto_eval_v0.csv")
OUT_MD = Path("reports/phase_i/i1_ram_ridge_predictor_summary_v0.md")

TARGET_COLS = [
    "rho_slip_target",
    "rho_rough_target",
    "sigma_target",
]

ALPHAS = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

def ff(x, default=np.nan):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def read_rows():
    if not IN_CSV.exists():
        raise SystemExit(f"[TRACER] missing {IN_CSV}")
    rows = list(csv.DictReader(open(IN_CSV)))
    if not rows:
        raise SystemExit(f"[TRACER] empty {IN_CSV}")
    return rows

def read_manifest():
    if not IN_MANIFEST.exists():
        raise SystemExit(f"[TRACER] missing {IN_MANIFEST}")
    return json.loads(IN_MANIFEST.read_text())

def build_X(rows, feature_cols, means=None, stds=None):
    X = []
    for r in rows:
        X.append([ff(r.get(c), np.nan) for c in feature_cols])
    X = np.asarray(X, dtype=float)

    if means is None:
        means = np.nanmean(X, axis=0)
        means = np.nan_to_num(means, nan=0.0)

    inds = np.where(~np.isfinite(X))
    if len(inds[0]) > 0:
        X[inds] = np.take(means, inds[1])

    if stds is None:
        stds = X.std(axis=0)
        stds = np.where(stds < 1e-8, 1.0, stds)

    Xn = (X - means) / stds
    return Xn, means, stds

def build_Y(rows):
    Y = []
    for r in rows:
        vals = [ff(r.get(c), np.nan) for c in TARGET_COLS]
        if any(not math.isfinite(float(v)) for v in vals):
            raise SystemExit(f"[TRACER] invalid target row: {r}")
        Y.append(vals)
    return np.asarray(Y, dtype=float)

def fit_ridge(X, Y, alpha):
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    reg = np.eye(Xb.shape[1]) * alpha
    reg[0, 0] = 0.0
    W = np.linalg.pinv(Xb.T @ Xb + reg) @ Xb.T @ Y
    return W

def predict(W, X):
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    raw = Xb @ W
    pred = clamp01(raw)
    return raw, pred

def l1(y, p):
    return float(np.abs(y - p).sum())

def rmse(y, p):
    return np.sqrt(np.mean((y - p) ** 2, axis=0))

def mean_predictor_eval(rows):
    base_tags = sorted(set(r["base_tag"] for r in rows))
    records = []
    all_l1 = []

    for heldout in base_tags:
        tr = [r for r in rows if r["base_tag"] != heldout]
        te = [r for r in rows if r["base_tag"] == heldout]

        Ytr = build_Y(tr)
        Yte = build_Y(te)
        pred = np.repeat(Ytr.mean(axis=0, keepdims=True), len(te), axis=0)

        row_l1 = [l1(Yte[i], pred[i]) for i in range(len(te))]
        all_l1.extend(row_l1)

        records.append({
            "heldout": heldout,
            "n": len(te),
            "mean_l1": float(np.mean(row_l1)),
            "target_mean": tuple(float(x) for x in Yte.mean(axis=0)),
            "pred_mean": tuple(float(x) for x in pred.mean(axis=0)),
        })

    return {
        "mean_loto_l1": float(np.mean(all_l1)),
        "max_loto_l1": float(np.max(all_l1)),
        "worst": max(records, key=lambda r: r["mean_l1"])["heldout"],
        "records": records,
    }

def evaluate_ridge(rows, feature_cols):
    base_tags = sorted(set(r["base_tag"] for r in rows))
    alpha_summaries = []

    for alpha in ALPHAS:
        records = []
        all_l1 = []

        for heldout in base_tags:
            tr = [r for r in rows if r["base_tag"] != heldout]
            te = [r for r in rows if r["base_tag"] == heldout]

            Xtr, means, stds = build_X(tr, feature_cols)
            Ytr = build_Y(tr)
            W = fit_ridge(Xtr, Ytr, alpha)

            Xte, _, _ = build_X(te, feature_cols, means=means, stds=stds)
            Yte = build_Y(te)
            raw, pred = predict(W, Xte)

            row_l1 = [l1(Yte[i], pred[i]) for i in range(len(te))]
            all_l1.extend(row_l1)

            r = rmse(Yte, pred)
            records.append({
                "heldout": heldout,
                "n": len(te),
                "mean_l1": float(np.mean(row_l1)),
                "rmse_slip": float(r[0]),
                "rmse_rough": float(r[1]),
                "rmse_sigma": float(r[2]),
                "target_mean": tuple(float(x) for x in Yte.mean(axis=0)),
                "pred_mean": tuple(float(x) for x in pred.mean(axis=0)),
            })

        alpha_summaries.append({
            "alpha": float(alpha),
            "mean_loto_l1": float(np.mean(all_l1)),
            "max_loto_l1": float(np.max(all_l1)),
            "worst": max(records, key=lambda r: r["mean_l1"])["heldout"],
            "records": records,
        })

    best = min(alpha_summaries, key=lambda r: (r["mean_loto_l1"], r["max_loto_l1"], r["alpha"]))

    Xall, means, stds = build_X(rows, feature_cols)
    Yall = build_Y(rows)
    W = fit_ridge(Xall, Yall, best["alpha"])
    raw_all, pred_all = predict(W, Xall)

    train_l1s = [l1(Yall[i], pred_all[i]) for i in range(len(rows))]
    train_rmse = rmse(Yall, pred_all)

    model = {
        "model_type": "standardized_linear_ridge_clamp01",
        "alpha": float(best["alpha"]),
        "feature_cols": feature_cols,
        "feature_means": [float(x) for x in means],
        "feature_stds": [float(x) for x in stds],
        "target_cols": TARGET_COLS,
        "coef_with_intercept": W.tolist(),
        "prediction_postprocess": "clamp each RAM output to [0,1]",
        "train_mean_l1": float(np.mean(train_l1s)),
        "train_max_l1": float(np.max(train_l1s)),
        "train_rmse": [float(x) for x in train_rmse],
    }

    return best, alpha_summaries, model

def fmt_vec(v):
    return "(" + ", ".join(f"{float(x):.4f}" for x in v) + ")"

rows = read_rows()
manifest = read_manifest()

feature_cols = [c for c in manifest["input_features"] if c in rows[0]]
if not feature_cols:
    raise SystemExit("[TRACER] no RAM input features found")

baseline = mean_predictor_eval(rows)
best, alpha_summaries, model = evaluate_ridge(rows, feature_cols)

OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
OUT_MODEL.write_text(json.dumps({
    "phase": "I1",
    "purpose": "RAM supervised pretraining ridge predictor",
    "input_csv": str(IN_CSV),
    "input_manifest": str(IN_MANIFEST),
    "safe_claim": "Predicts heuristic RAM teacher seeds; not final teacher-student RAM.",
    "baseline_mean_predictor": {
        "mean_loto_l1": baseline["mean_loto_l1"],
        "max_loto_l1": baseline["max_loto_l1"],
        "worst": baseline["worst"],
    },
    "ridge_loto": {
        "best_alpha": best["alpha"],
        "mean_loto_l1": best["mean_loto_l1"],
        "max_loto_l1": best["max_loto_l1"],
        "worst": best["worst"],
    },
    "alpha_summary": [
        {
            "alpha": a["alpha"],
            "mean_loto_l1": a["mean_loto_l1"],
            "max_loto_l1": a["max_loto_l1"],
            "worst": a["worst"],
        }
        for a in alpha_summaries
    ],
    "model": model,
}, indent=2))

OUT_EVAL.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_EVAL, "w", newline="") as f:
    fields = [
        "model",
        "heldout_base_tag",
        "n",
        "mean_l1",
        "rmse_slip",
        "rmse_rough",
        "rmse_sigma",
        "target_mean",
        "pred_mean",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()

    for r in baseline["records"]:
        w.writerow({
            "model": "mean_baseline",
            "heldout_base_tag": r["heldout"],
            "n": r["n"],
            "mean_l1": r["mean_l1"],
            "rmse_slip": "",
            "rmse_rough": "",
            "rmse_sigma": "",
            "target_mean": fmt_vec(r["target_mean"]),
            "pred_mean": fmt_vec(r["pred_mean"]),
        })

    for r in best["records"]:
        w.writerow({
            "model": "ridge",
            "heldout_base_tag": r["heldout"],
            "n": r["n"],
            "mean_l1": r["mean_l1"],
            "rmse_slip": r["rmse_slip"],
            "rmse_rough": r["rmse_rough"],
            "rmse_sigma": r["rmse_sigma"],
            "target_mean": fmt_vec(r["target_mean"]),
            "pred_mean": fmt_vec(r["pred_mean"]),
        })

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-I1 RAM Ridge Predictor v0\n\n")
    f.write("This trains a supervised RAM predictor on the Phase-I0 heuristic RAM teacher dataset.\n\n")
    f.write(f"- input csv: `{IN_CSV}`\n")
    f.write(f"- manifest: `{IN_MANIFEST}`\n")
    f.write(f"- model: `{OUT_MODEL}`\n")
    f.write(f"- eval csv: `{OUT_EVAL}`\n")
    f.write(f"- rows: `{len(rows)}`\n")
    f.write(f"- features: `{len(feature_cols)}`\n")
    f.write("- targets: `rho_slip_target`, `rho_rough_target`, `sigma_target`\n\n")

    f.write("## Summary\n\n")
    f.write("| model | best alpha | mean LOTO L1 | max LOTO L1 | worst heldout | train mean L1 | train max L1 |\n")
    f.write("|---|---:|---:|---:|---|---:|---:|\n")
    f.write(
        f"| mean_baseline | NA | {baseline['mean_loto_l1']:.6f} | {baseline['max_loto_l1']:.6f} | "
        f"{baseline['worst']} | NA | NA |\n"
    )
    f.write(
        f"| ridge | {best['alpha']:.6g} | {best['mean_loto_l1']:.6f} | {best['max_loto_l1']:.6f} | "
        f"{best['worst']} | {model['train_mean_l1']:.6f} | {model['train_max_l1']:.6f} |\n"
    )

    f.write("\n## Best ridge leave-one-base-tag-out detail\n\n")
    f.write("| heldout | n | mean L1 | RMSE slip | RMSE rough | RMSE sigma | target mean | pred mean |\n")
    f.write("|---|---:|---:|---:|---:|---:|---|---|\n")
    for r in best["records"]:
        f.write(
            f"| {r['heldout']} | {r['n']} | {r['mean_l1']:.6f} | "
            f"{r['rmse_slip']:.6f} | {r['rmse_rough']:.6f} | {r['rmse_sigma']:.6f} | "
            f"{fmt_vec(r['target_mean'])} | {fmt_vec(r['pred_mean'])} |\n"
        )

    f.write("\n## Interpretation\n\n")
    f.write("- This is a RAM supervised pretraining baseline from heuristic teacher seeds.\n")
    f.write("- It is not yet the final teacher-student RAM trained from proprioceptive prediction error.\n")
    f.write("- If ridge improves over the mean baseline, the I0 teacher signals are at least partially predictable from runtime-style features.\n")
    f.write("- Sparse anchors such as p045 should remain high-uncertainty and should not be overfit.\n")

print(f"[TRACER] wrote {OUT_MODEL}")
print(f"[TRACER] wrote {OUT_EVAL}")
print(f"[TRACER] wrote {OUT_MD}")
print(
    f"[TRACER] mean_baseline: mean_loto_l1={baseline['mean_loto_l1']:.6f} "
    f"max={baseline['max_loto_l1']:.6f} worst={baseline['worst']}"
)
print(
    f"[TRACER] ridge: alpha={best['alpha']} mean_loto_l1={best['mean_loto_l1']:.6f} "
    f"max={best['max_loto_l1']:.6f} worst={best['worst']}"
)
