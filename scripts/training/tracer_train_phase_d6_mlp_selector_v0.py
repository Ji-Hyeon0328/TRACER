#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn


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


class MLPSelector(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def tensor_stats(x):
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean, std


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


def fmt(x):
    return "NA" if x is None else f"{x:.8f}"


@torch.no_grad()
def predict_original(model, X, x_mean, x_std, y_mean, y_std):
    Xn = (X - x_mean) / x_std
    Yn = model(Xn)
    return Yn * y_std + y_mean


def compute_metrics(Y_pred, Y_true):
    out = {}
    err = Y_pred - Y_true
    for j, target in enumerate(TARGETS):
        vals = err[:, j].detach().cpu().tolist()
        out[target] = {
            "rmse": rmse(vals),
            "mae": mae(vals),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-pt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-pred-csv", required=True)
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--train-ratio", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = Path(args.dataset)
    out_pt = Path(args.out_pt)
    out_json = Path(args.out_json)
    out_summary = Path(args.out_summary)
    out_pred_csv = Path(args.out_pred_csv)

    for p in [out_pt, out_json, out_summary, out_pred_csv]:
        p.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(dataset.open()))
    if not rows:
        raise SystemExit(f"[ERROR] empty dataset: {dataset}")

    X_list = [[fnum(r.get(c)) for c in FEATURES] for r in rows]
    Y_list = [[fnum(r.get(t)) for t in TARGETS] for r in rows]
    contexts = [r.get("context", "unknown") for r in rows]

    X = torch.tensor(X_list, dtype=torch.float32)
    Y = torch.tensor(Y_list, dtype=torch.float32)

    n = len(rows)
    indices = list(range(n))
    random.shuffle(indices)

    n_train = max(1, int(n * args.train_ratio))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:] if n_train < n else indices[:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    x_mean, x_std = tensor_stats(X_train)
    y_mean, y_std = tensor_stats(Y_train)

    X_train_n = (X_train - x_mean) / x_std
    Y_train_n = (Y_train - y_mean) / y_std

    hidden_sizes = [int(v) for v in args.hidden.split(",") if v.strip()]
    model = MLPSelector(len(FEATURES), len(TARGETS), hidden_sizes)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(X_train_n)
        loss = loss_fn(pred, Y_train_n)
        loss.backward()
        opt.step()

        loss_val = float(loss.item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    Y_pred_train = predict_original(model, X_train, x_mean, x_std, y_mean, y_std)
    Y_pred_test = predict_original(model, X_test, x_mean, x_std, y_mean, y_std)
    Y_pred_all = predict_original(model, X, x_mean, x_std, y_mean, y_std)

    train_metrics = compute_metrics(Y_pred_train, Y_train)
    test_metrics = compute_metrics(Y_pred_test, Y_test)
    all_metrics = compute_metrics(Y_pred_all, Y)

    context_metrics = {}
    for ctx in sorted(set(contexts)):
        idxs = [i for i in test_idx if contexts[i] == ctx]
        if not idxs:
            continue
        yp = Y_pred_all[idxs]
        yt = Y[idxs]
        context_metrics[ctx] = {
            "n": len(idxs),
            "metrics": compute_metrics(yp, yt),
        }

    checkpoint = {
        "state_dict": model.state_dict(),
        "features": FEATURES,
        "targets": TARGETS,
        "hidden_sizes": hidden_sizes,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    torch.save(checkpoint, out_pt)

    metadata = {
        "model_name": "phase_d6_mlp_selector_v0",
        "model_type": "supervised_mlp_selector",
        "source_dataset": str(dataset),
        "out_pt": str(out_pt),
        "features": FEATURES,
        "targets": TARGETS,
        "hidden_sizes": hidden_sizes,
        "epochs": args.epochs,
        "best_train_normalized_mse": best_loss,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "num_samples": n,
        "num_train": len(train_idx),
        "num_test": len(test_idx),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "all_metrics": all_metrics,
        "context_metrics": context_metrics,
        "note": (
            "Offline supervised MLP selector trained to imitate Phase-D5 empirical/gated references. "
            "This is not yet an RL policy."
        ),
    }
    out_json.write_text(json.dumps(metadata, indent=2))

    with out_pred_csv.open("w", newline="") as f:
        fields = ["idx", "split", "trial", "context", "x", "y"]
        for t in TARGETS:
            fields += [t, f"pred_{t}", f"err_{t}"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        train_set = set(train_idx)
        for i, r in enumerate(rows):
            out = {
                "idx": i,
                "split": "train" if i in train_set else "test",
                "trial": r.get("trial"),
                "context": r.get("context"),
                "x": r.get("x"),
                "y": r.get("y"),
            }
            for j, t in enumerate(TARGETS):
                y = float(Y[i, j].item())
                yp = float(Y_pred_all[i, j].item())
                out[t] = y
                out[f"pred_{t}"] = yp
                out[f"err_{t}"] = yp - y
            w.writerow(out)

    lines = []
    lines.append("# TRACER Phase-D6 MLP Selector v0")
    lines.append("")
    lines.append(f"- model_type: `{metadata['model_type']}`")
    lines.append(f"- source_dataset: `{dataset}`")
    lines.append(f"- out_pt: `{out_pt}`")
    lines.append(f"- out_json: `{out_json}`")
    lines.append(f"- out_pred_csv: `{out_pred_csv}`")
    lines.append(f"- num_samples: `{n}`")
    lines.append(f"- num_train: `{len(train_idx)}`")
    lines.append(f"- num_test: `{len(test_idx)}`")
    lines.append(f"- hidden_sizes: `{hidden_sizes}`")
    lines.append(f"- epochs: `{args.epochs}`")
    lines.append(f"- best_train_normalized_mse: `{best_loss}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| target | train_rmse | train_mae | test_rmse | test_mae | all_rmse | all_mae |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for t in TARGETS:
        tm = train_metrics[t]
        vm = test_metrics[t]
        am = all_metrics[t]
        lines.append(
            f"| {t} | "
            f"{fmt(tm['rmse'])} | {fmt(tm['mae'])} | "
            f"{fmt(vm['rmse'])} | {fmt(vm['mae'])} | "
            f"{fmt(am['rmse'])} | {fmt(am['mae'])} |"
        )

    lines.append("")
    lines.append("## Context-wise test metrics")
    lines.append("")
    lines.append("| context | n | vx_rmse | yaw_rmse | body_h_rmse | clearance_rmse | enable_rmse |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ctx, cm in context_metrics.items():
        m = cm["metrics"]
        lines.append(
            f"| {ctx} | {cm['n']} | "
            f"{fmt(m['target_vx']['rmse'])} | "
            f"{fmt(m['target_yaw_rate']['rmse'])} | "
            f"{fmt(m['target_body_h']['rmse'])} | "
            f"{fmt(m['target_clearance']['rmse'])} | "
            f"{fmt(m['target_enable']['rmse'])} |"
        )

    out_summary.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] wrote {out_pt}")
    print(f"[TRACER] wrote {out_json}")
    print(f"[TRACER] wrote {out_summary}")
    print(f"[TRACER] wrote {out_pred_csv}")
    print("[TRACER] test metrics:")
    for t in TARGETS:
        print(
            f"  {t}: "
            f"rmse={fmt(test_metrics[t]['rmse'])}, "
            f"mae={fmt(test_metrics[t]['mae'])}"
        )


if __name__ == "__main__":
    main()
