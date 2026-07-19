#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


FEATURES = [
    "ctx_flat",
    "ctx_start_flat",
    "ctx_upslope",
    "ctx_rough",
    "ctx_downslope",
    "ctx_goal_flat",
    "ctx_unknown",
    "x",
    "y",
    "beta_motion",
    "beta_stability",
    "beta_energy",
    "ram_slip_proxy",
    "ram_roughness_proxy",
    "ram_sigma",
]

TARGETS = [
    "target_vx",
    "target_yaw_rate",
    "target_body_h",
    "target_clearance",
    "target_enable",
]


def fnum(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def mat_transpose(A):
    return [list(row) for row in zip(*A)]


def matmul(A, B):
    Bt = mat_transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def matvec(A, x):
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def solve_linear(A, b):
    # Gaussian elimination with partial pivoting.
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            M[pivot][col] = 1e-12
        M[col], M[pivot] = M[pivot], M[col]

        div = M[col][col]
        M[col] = [v / div for v in M[col]]

        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor == 0:
                continue
            M[r] = [rv - factor * cv for rv, cv in zip(M[r], M[col])]

    return [M[i][-1] for i in range(n)]


def fit_ridge(X, y, alpha):
    # Adds bias internally by appending 1.0 to each row.
    Xb = [row + [1.0] for row in X]
    Xt = mat_transpose(Xb)
    XtX = matmul(Xt, Xb)
    Xty = matvec(Xt, y)

    n = len(XtX)
    for i in range(n):
        XtX[i][i] += alpha

    # Do not regularize bias strongly.
    XtX[-1][-1] -= alpha
    XtX[-1][-1] += alpha * 1e-6

    w = solve_linear(XtX, Xty)
    return w[:-1], w[-1]


def predict_row(row, weights, bias):
    return sum(a * b for a, b in zip(row, weights)) + bias


def rmse(vals):
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return None
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def mae(vals):
    vals = [abs(v) for v in vals if math.isfinite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--train-ratio", type=float, default=0.80)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    out_json = Path(args.out_json)
    out_summary = Path(args.out_summary)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(dataset.open()))
    if not rows:
        raise SystemExit(f"[ERROR] empty dataset: {dataset}")

    X = [[fnum(r.get(c)) for c in FEATURES] for r in rows]
    Y = {t: [fnum(r.get(t)) for r in rows] for t in TARGETS}
    contexts = [r.get("context", "unknown") for r in rows]

    n = len(rows)
    n_train = max(1, int(n * args.train_ratio))

    X_train = X[:n_train]
    X_test = X[n_train:] if n_train < n else X[:]

    models = {}
    metrics = {}

    for target in TARGETS:
        y_train = Y[target][:n_train]
        y_test = Y[target][n_train:] if n_train < n else Y[target][:]

        w, b = fit_ridge(X_train, y_train, args.alpha)

        pred_train = [predict_row(x, w, b) for x in X_train]
        pred_test = [predict_row(x, w, b) for x in X_test]

        train_err = [p - y for p, y in zip(pred_train, y_train)]
        test_err = [p - y for p, y in zip(pred_test, y_test)]

        models[target] = {
            "weights": w,
            "bias": b,
        }
        metrics[target] = {
            "train_rmse": rmse(train_err),
            "train_mae": mae(train_err),
            "test_rmse": rmse(test_err),
            "test_mae": mae(test_err),
        }

    # Context-wise test metrics.
    context_metrics = {}
    test_indices = list(range(n_train, n)) if n_train < n else list(range(n))
    for ctx in sorted(set(contexts)):
        idxs = [i for i in test_indices if contexts[i] == ctx]
        if not idxs:
            continue
        context_metrics[ctx] = {"n": len(idxs)}
        for target in TARGETS:
            w = models[target]["weights"]
            b = models[target]["bias"]
            errs = [predict_row(X[i], w, b) - Y[target][i] for i in idxs]
            context_metrics[ctx][f"{target}_rmse"] = rmse(errs)
            context_metrics[ctx][f"{target}_mae"] = mae(errs)

    model = {
        "model_name": "phase_d5_learned_selector_ridge_v0",
        "model_type": "ridge_regression_multi_output",
        "source_dataset": str(dataset),
        "features": FEATURES,
        "targets": TARGETS,
        "alpha": args.alpha,
        "train_ratio": args.train_ratio,
        "num_samples": n,
        "num_train": len(X_train),
        "num_test": len(X_test),
        "models": models,
        "metrics": metrics,
        "context_metrics": context_metrics,
        "note": (
            "Offline shadow learned selector. This model imitates the Phase-D5 "
            "empirical policy targets and is not yet used for runtime control."
        ),
    }

    out_json.write_text(json.dumps(model, indent=2))

    lines = []
    lines.append("# TRACER Phase-D5 Learned Selector v0")
    lines.append("")
    lines.append(f"- model_type: `{model['model_type']}`")
    lines.append(f"- source_dataset: `{dataset}`")
    lines.append(f"- out_json: `{out_json}`")
    lines.append(f"- num_samples: `{n}`")
    lines.append(f"- num_train: `{len(X_train)}`")
    lines.append(f"- num_test: `{len(X_test)}`")
    lines.append(f"- alpha: `{args.alpha}`")
    lines.append("")
    lines.append("## Overall metrics")
    lines.append("")
    lines.append("| target | train_rmse | train_mae | test_rmse | test_mae |")
    lines.append("|---|---:|---:|---:|---:|")
    for t in TARGETS:
        m = metrics[t]
        lines.append(
            f"| {t} | "
            f"{m['train_rmse']:.8f} | {m['train_mae']:.8f} | "
            f"{m['test_rmse']:.8f} | {m['test_mae']:.8f} |"
        )

    lines.append("")
    lines.append("## Context-wise test metrics")
    lines.append("")
    lines.append("| context | n | vx_rmse | yaw_rmse | body_h_rmse | clearance_rmse | enable_rmse |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ctx, cm in context_metrics.items():
        lines.append(
            f"| {ctx} | {cm['n']} | "
            f"{cm.get('target_vx_rmse', 0.0):.8f} | "
            f"{cm.get('target_yaw_rate_rmse', 0.0):.8f} | "
            f"{cm.get('target_body_h_rmse', 0.0):.8f} | "
            f"{cm.get('target_clearance_rmse', 0.0):.8f} | "
            f"{cm.get('target_enable_rmse', 0.0):.8f} |"
        )

    out_summary.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] wrote {out_json}")
    print(f"[TRACER] wrote {out_summary}")
    print("[TRACER] metrics:")
    for t in TARGETS:
        print(f"  {t}: test_rmse={metrics[t]['test_rmse']:.8f}, test_mae={metrics[t]['test_mae']:.8f}")


if __name__ == "__main__":
    main()
