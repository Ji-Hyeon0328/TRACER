#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

IN_CSV = Path("datasets/phase_h/h0_objective_selector_training_table_v0.csv")
IN_MANIFEST = Path("datasets/phase_h/h0_objective_selector_feature_manifest_v0.json")

OUT_MODEL = Path("models/phase_h/h1_objective_selector_ridge_v0.json")
OUT_EVAL = Path("datasets/phase_h/h1_objective_selector_loto_eval_v0.csv")
OUT_MD = Path("reports/phase_h/h1_objective_selector_ridge_summary_v0.md")

TARGET_COLS = [
    "beta_motion_target",
    "beta_stability_target",
    "beta_energy_target",
]

ALPHAS = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

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

def project_simplex(v, eps=1e-8):
    """
    Euclidean projection to probability simplex.
    """
    v = np.asarray(v, dtype=float)
    if np.all(np.isfinite(v)):
        pass
    else:
        v = np.nan_to_num(v, nan=1.0 / len(v), posinf=1.0, neginf=0.0)

    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0

    if not np.any(cond):
        return np.ones(n) / n

    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0.0)

    s = float(w.sum())
    if s <= eps:
        return np.ones(n) / n
    return w / s

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

def build_feature_matrix(rows, feature_cols, means=None, stds=None):
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

def build_target_matrix(rows):
    Y = []
    for r in rows:
        vals = [ff(r.get(c), np.nan) for c in TARGET_COLS]
        if any(not math.isfinite(float(v)) for v in vals):
            raise SystemExit(f"[TRACER] invalid target row: {r}")
        vals = project_simplex(vals)
        Y.append(vals)
    return np.asarray(Y, dtype=float)

def fit_ridge(X, Y, alpha):
    """
    Fit Y = [1 X] W.
    Do not regularize intercept.
    """
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    reg = np.eye(Xb.shape[1]) * alpha
    reg[0, 0] = 0.0
    W = np.linalg.pinv(Xb.T @ Xb + reg) @ Xb.T @ Y
    return W

def predict(W, X):
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    raw = Xb @ W
    proj = np.asarray([project_simplex(r) for r in raw], dtype=float)
    return raw, proj

def beta_l1(y, p):
    return float(np.abs(y - p).sum())

def rmse(y, p):
    return np.sqrt(np.mean((y - p) ** 2, axis=0))

def evaluate_suite(rows, feature_cols, suite_name):
    base_tags = sorted(set(r["base_tag"] for r in rows))
    eval_records = []
    alpha_summary = []

    for alpha in ALPHAS:
        l1s = []
        heldout_records = []

        for heldout in base_tags:
            train_rows = [r for r in rows if r["base_tag"] != heldout]
            test_rows = [r for r in rows if r["base_tag"] == heldout]

            Xtr, means, stds = build_feature_matrix(train_rows, feature_cols)
            Ytr = build_target_matrix(train_rows)
            W = fit_ridge(Xtr, Ytr, alpha)

            Xte, _, _ = build_feature_matrix(test_rows, feature_cols, means=means, stds=stds)
            Yte = build_target_matrix(test_rows)
            raw, pred = predict(W, Xte)

            row_l1s = [beta_l1(Yte[i], pred[i]) for i in range(len(test_rows))]
            mean_l1 = float(np.mean(row_l1s))
            l1s.extend(row_l1s)

            heldout_records.append({
                "suite": suite_name,
                "alpha": alpha,
                "heldout_base_tag": heldout,
                "n_test": len(test_rows),
                "mean_l1": mean_l1,
                "rmse_motion": float(rmse(Yte, pred)[0]),
                "rmse_stability": float(rmse(Yte, pred)[1]),
                "rmse_energy": float(rmse(Yte, pred)[2]),
                "target_mean": tuple(float(x) for x in Yte.mean(axis=0)),
                "pred_mean": tuple(float(x) for x in pred.mean(axis=0)),
            })

        alpha_summary.append({
            "suite": suite_name,
            "alpha": alpha,
            "mean_loto_l1": float(np.mean(l1s)),
            "max_loto_l1": float(np.max(l1s)),
            "worst_base_tag": max(heldout_records, key=lambda r: r["mean_l1"])["heldout_base_tag"],
            "records": heldout_records,
        })

    best = min(alpha_summary, key=lambda r: (r["mean_loto_l1"], r["max_loto_l1"], r["alpha"]))

    # Fit final full model using best alpha.
    Xall, means, stds = build_feature_matrix(rows, feature_cols)
    Yall = build_target_matrix(rows)
    W = fit_ridge(Xall, Yall, best["alpha"])
    raw_all, pred_all = predict(W, Xall)

    train_l1s = [beta_l1(Yall[i], pred_all[i]) for i in range(len(rows))]
    train_rmse = rmse(Yall, pred_all)

    final_model = {
        "suite": suite_name,
        "model_type": "standardized_linear_ridge_simplex_projection",
        "alpha": float(best["alpha"]),
        "feature_cols": feature_cols,
        "feature_means": [float(x) for x in means],
        "feature_stds": [float(x) for x in stds],
        "target_cols": TARGET_COLS,
        "coef_with_intercept": W.tolist(),
        "prediction_postprocess": "project raw 3-vector to probability simplex",
        "train_mean_l1": float(np.mean(train_l1s)),
        "train_max_l1": float(np.max(train_l1s)),
        "train_rmse": [float(x) for x in train_rmse],
    }

    return best, alpha_summary, final_model

def fmt_beta(v):
    return "(" + ", ".join(f"{float(x):.4f}" for x in v) + ")"

rows = read_rows()
manifest = read_manifest()

deployable_features = manifest["feature_groups"]["deployable_pretrain"]
diagnostic_features = manifest["feature_groups"]["diagnostic_only"]

suites = {
    "deployable_pretrain": deployable_features,
    "diagnostic_with_reset_y": deployable_features + diagnostic_features,
}

results = {}
all_eval_rows = []

for suite_name, feature_cols in suites.items():
    feature_cols = [c for c in feature_cols if c in rows[0]]
    if not feature_cols:
        raise SystemExit(f"[TRACER] no features for suite {suite_name}")

    best, alpha_summary, model = evaluate_suite(rows, feature_cols, suite_name)

    results[suite_name] = {
        "best_alpha": float(best["alpha"]),
        "best_mean_loto_l1": float(best["mean_loto_l1"]),
        "best_max_loto_l1": float(best["max_loto_l1"]),
        "best_worst_base_tag": best["worst_base_tag"],
        "n_features": len(feature_cols),
        "alpha_summary": [
            {
                "alpha": float(a["alpha"]),
                "mean_loto_l1": float(a["mean_loto_l1"]),
                "max_loto_l1": float(a["max_loto_l1"]),
                "worst_base_tag": a["worst_base_tag"],
            }
            for a in alpha_summary
        ],
        "model": model,
    }

    for rec in best["records"]:
        all_eval_rows.append({
            "suite": rec["suite"],
            "alpha": rec["alpha"],
            "heldout_base_tag": rec["heldout_base_tag"],
            "n_test": rec["n_test"],
            "mean_l1": rec["mean_l1"],
            "rmse_motion": rec["rmse_motion"],
            "rmse_stability": rec["rmse_stability"],
            "rmse_energy": rec["rmse_energy"],
            "target_mean": fmt_beta(rec["target_mean"]),
            "pred_mean": fmt_beta(rec["pred_mean"]),
        })

OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
OUT_MODEL.write_text(json.dumps({
    "phase": "H1",
    "purpose": "Objective Selector beta supervised pretraining ridge baseline",
    "input_csv": str(IN_CSV),
    "input_manifest": str(IN_MANIFEST),
    "safe_claim": "Supervised beta pretraining on robust true-metric teacher seeds; not final IRL.",
    "suites": results,
}, indent=2))

OUT_EVAL.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_EVAL, "w", newline="") as f:
    fields = [
        "suite",
        "alpha",
        "heldout_base_tag",
        "n_test",
        "mean_l1",
        "rmse_motion",
        "rmse_stability",
        "rmse_energy",
        "target_mean",
        "pred_mean",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(all_eval_rows)

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-H1 Objective Selector Ridge v0\n\n")
    f.write("This trains ridge-regression Objective Selector beta baselines on the H0 supervised pretraining table.\n\n")
    f.write(f"- input csv: `{IN_CSV}`\n")
    f.write(f"- model: `{OUT_MODEL}`\n")
    f.write(f"- LOTO eval csv: `{OUT_EVAL}`\n")
    f.write(f"- rows: `{len(rows)}`\n")
    f.write("- target: robust true-metric teacher seed beta from H0\n")
    f.write("- postprocess: raw beta prediction projected to probability simplex\n\n")

    f.write("## Suite summary\n\n")
    f.write("| suite | features | best alpha | mean LOTO L1 | max LOTO L1 | worst heldout | train mean L1 | train max L1 |\n")
    f.write("|---|---:|---:|---:|---:|---|---:|---:|\n")
    for suite_name, res in results.items():
        m = res["model"]
        f.write(
            f"| {suite_name} | {res['n_features']} | {res['best_alpha']:.6g} | "
            f"{res['best_mean_loto_l1']:.6f} | {res['best_max_loto_l1']:.6f} | "
            f"{res['best_worst_base_tag']} | {m['train_mean_l1']:.6f} | {m['train_max_l1']:.6f} |\n"
        )

    f.write("\n## Best-alpha leave-one-base-tag-out detail\n\n")
    f.write("| suite | heldout | n | mean L1 | RMSE motion | RMSE stability | RMSE energy | target mean | pred mean |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---|---|\n")
    for r in all_eval_rows:
        f.write(
            f"| {r['suite']} | {r['heldout_base_tag']} | {r['n_test']} | "
            f"{float(r['mean_l1']):.6f} | {float(r['rmse_motion']):.6f} | "
            f"{float(r['rmse_stability']):.6f} | {float(r['rmse_energy']):.6f} | "
            f"{r['target_mean']} | {r['pred_mean']} |\n"
        )

    f.write("\n## Interpretation\n\n")
    f.write("- `deployable_pretrain` excludes artificial reset-y diagnostic features.\n")
    f.write("- `diagnostic_with_reset_y` includes reset-y features and is useful for checking separability, but should not be treated as a deployable robot input.\n")
    f.write("- This is still supervised pretraining from robust true-metric teacher seeds, not final IRL.\n")
    f.write("- If LOTO remains weak for p060/p045, the bottleneck is condition coverage and RAM/context representation rather than just the regressor.\n")

print(f"[TRACER] wrote {OUT_MODEL}")
print(f"[TRACER] wrote {OUT_EVAL}")
print(f"[TRACER] wrote {OUT_MD}")
for suite_name, res in results.items():
    print(
        f"[TRACER] {suite_name}: features={res['n_features']} "
        f"alpha={res['best_alpha']} mean_loto_l1={res['best_mean_loto_l1']:.6f} "
        f"max={res['best_max_loto_l1']:.6f} worst={res['best_worst_base_tag']}"
    )
