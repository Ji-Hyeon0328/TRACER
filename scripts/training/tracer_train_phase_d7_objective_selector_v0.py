#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from pathlib import Path


CONTEXTS = ["flat", "start_flat", "upslope", "rough", "downslope", "goal_flat", "unknown"]


FEATURES = [
    # context one-hot is appended first
    "reset_y",
    "y_mean",
    "mean_abs_y_context",
    "max_abs_y_context",
    "ram_slip_proxy_mean",
    "ram_roughness_proxy_mean",
    "ram_sigma_mean",
    "motion_score",
    "stability_score",
    "energy_score",
    "effort_proxy",
    "hold_drift",
    "rollout_max_abs_y",
    "rollout_mean_abs_y",
]


TARGETS = [
    "target_beta_motion",
    "target_beta_stability",
    "target_beta_energy",
]


def f(row, key, default=0.0):
    try:
        v = row.get(key, default)
        if v in ("", None):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def load_rows(path):
    with Path(path).open(newline="") as fp:
        return list(csv.DictReader(fp))


def make_feature(row):
    ctx = row.get("context", "unknown")
    x = [1.0 if ctx == c else 0.0 for c in CONTEXTS]
    x += [f(row, k) for k in FEATURES]
    return x


def make_target(row):
    return [f(row, k) for k in TARGETS]


def normalize_beta(v):
    v = [max(1e-9, float(x)) for x in v]
    s = sum(v)
    return [x / s for x in v]


def mat_transpose(A):
    return list(map(list, zip(*A)))


def matmul(A, B):
    n = len(A)
    m = len(B[0])
    k = len(B)
    out = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for t in range(k):
            a = Ai[t]
            if a == 0:
                continue
            Bt = B[t]
            for j in range(m):
                out[i][j] += a * Bt[j]
    return out


def solve_linear(A, B):
    # Gaussian elimination with partial pivot.
    n = len(A)
    m = len(B[0])
    M = [list(A[i]) + list(B[i]) for i in range(n)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise RuntimeError("singular matrix")
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]

        div = M[col][col]
        for j in range(col, n + m):
            M[col][j] /= div

        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if abs(factor) < 1e-18:
                continue
            for j in range(col, n + m):
                M[r][j] -= factor * M[col][j]

    return [row[n:] for row in M]


def fit_ridge(X, Y, alpha):
    # Add intercept column.
    Xb = [[1.0] + list(x) for x in X]
    Xt = mat_transpose(Xb)
    XtX = matmul(Xt, Xb)
    XtY = matmul(Xt, Y)

    for i in range(len(XtX)):
        XtX[i][i] += alpha

    # Do not regularize intercept.
    XtX[0][0] -= alpha

    W = solve_linear(XtX, XtY)
    return W


def predict(W, x):
    xb = [1.0] + list(x)
    y = []
    for j in range(len(W[0])):
        yj = sum(xb[i] * W[i][j] for i in range(len(xb)))
        y.append(yj)
    return normalize_beta(y)


def rmse_mae(y_true, y_pred):
    n = len(y_true)
    d = len(y_true[0])
    out = {}
    for j, name in enumerate(TARGETS):
        se = sum((y_true[i][j] - y_pred[i][j]) ** 2 for i in range(n)) / max(1, n)
        ae = sum(abs(y_true[i][j] - y_pred[i][j]) for i in range(n)) / max(1, n)
        out[name] = {"rmse": math.sqrt(se), "mae": ae}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit("[ERROR] empty csv")

    X = [make_feature(r) for r in rows]
    Y = [make_target(r) for r in rows]
    Y = [normalize_beta(y) for y in Y]

    idx = list(range(len(rows)))
    rng = random.Random(args.seed)
    rng.shuffle(idx)

    n_train = max(1, int(len(idx) * args.train_ratio))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:] or idx[:]

    X_train = [X[i] for i in train_idx]
    Y_train = [Y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    Y_test = [Y[i] for i in test_idx]

    W = fit_ridge(X_train, Y_train, args.alpha)

    pred_train = [predict(W, x) for x in X_train]
    pred_test = [predict(W, x) for x in X_test]
    pred_all = [predict(W, x) for x in X]

    train_metrics = rmse_mae(Y_train, pred_train)
    test_metrics = rmse_mae(Y_test, pred_test)
    all_metrics = rmse_mae(Y, pred_all)

    feature_names = ["intercept"] + [f"ctx_{c}" for c in CONTEXTS] + FEATURES

    model = {
        "model_type": "ridge_regression_beta_selector_v0",
        "source_csv": args.csv,
        "contexts": CONTEXTS,
        "features": FEATURES,
        "feature_names_with_intercept": feature_names,
        "targets": TARGETS,
        "alpha": args.alpha,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "n_samples": len(rows),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "weights": W,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "all_metrics": all_metrics,
        "notes": [
            "This is a supervised bootstrap Objective Selector, not IRL yet.",
            "Predicted beta is clamped positive and normalized to sum to 1.",
            "Runtime use should prefer online-observable features; rollout-only scores are for D7 bootstrap analysis."
        ],
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(model, indent=2))

    lines = []
    lines.append("# TRACER Phase-D7.0c Objective Selector Training Summary v0")
    lines.append("")
    lines.append(f"- source_csv: `{args.csv}`")
    lines.append(f"- model_json: `{out_json}`")
    lines.append(f"- model_type: `ridge_regression_beta_selector_v0`")
    lines.append(f"- samples: `{len(rows)}`")
    lines.append(f"- train/test: `{len(train_idx)}` / `{len(test_idx)}`")
    lines.append(f"- alpha: `{args.alpha}`")
    lines.append("")
    lines.append("## Test metrics")
    lines.append("")
    lines.append("| target | rmse | mae |")
    lines.append("|---|---:|---:|")
    for t in TARGETS:
        lines.append(f"| {t} | {test_metrics[t]['rmse']:.6f} | {test_metrics[t]['mae']:.6f} |")
    lines.append("")
    lines.append("## All-sample metrics")
    lines.append("")
    lines.append("| target | rmse | mae |")
    lines.append("|---|---:|---:|")
    for t in TARGETS:
        lines.append(f"| {t} | {all_metrics[t]['rmse']:.6f} | {all_metrics[t]['mae']:.6f} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This is the first data-derived Objective Selector model.")
    lines.append("- It predicts beta targets from D7 bootstrap features.")
    lines.append("- It is not yet preference-based IRL or RL.")
    lines.append("- Next step is D7.0d: evaluate predictions by context/reset-y and prepare a runtime shadow node.")
    lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] wrote {out_json}")
    print(f"[TRACER] wrote {out_md}")
    print("[TRACER] test_metrics:")
    for t in TARGETS:
        print(f"  {t}: rmse={test_metrics[t]['rmse']:.6f}, mae={test_metrics[t]['mae']:.6f}")


if __name__ == "__main__":
    main()
