#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in ["true", "1", "yes"]


def ffloat(row, key, default=0.0):
    v = row.get(key, "")
    if v in ["", "NA", "None", None]:
        return default
    return float(v)


def parse_schedule(schedule):
    chunks = [c.strip() for c in schedule.split(";") if c.strip()]
    p0 = [x.strip() for x in chunks[0].split(",")]
    p1 = [x.strip() for x in chunks[1].split(",")]
    return {
        "vx_start": float(p0[2]),
        "body_h": float(p0[3]),
        "clearance": float(p0[4]),
        "vx_tail": float(p1[2]),
    }


def load_rollouts(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def objective_return(row, beta, goal_x, lateral_bound, timeout_s):
    a = parse_schedule(row["schedule"])

    success = parse_bool(row.get("success", False))
    goal = parse_bool(row.get("goal", False))
    out_lane = parse_bool(row.get("out_lane", False))
    startup_failed = parse_bool(row.get("startup_failed", False))

    final_x = ffloat(row, "final_x", 0.0)
    max_abs_y = ffloat(row, "max_abs_y", 0.0)
    mean_abs_y = ffloat(row, "mean_abs_y", 0.0)

    time_raw = row.get("time", "")
    time_s = timeout_s if time_raw in ["", "NA", "None", None] else float(time_raw)

    progress = max(0.0, min(final_x / goal_x, 1.0))
    time_eff = max(0.0, 1.0 - min(time_s / timeout_s, 1.0))

    vx_start_norm = (a["vx_start"] - 0.195) / (0.225 - 0.195)
    vx_tail_norm = (a["vx_tail"] - 0.190) / (0.210 - 0.190)
    h_dev = (a["body_h"] - 0.320) / max(0.324 - 0.318, 1e-9)
    c_dev = (a["clearance"] - 0.045) / max(0.050 - 0.043, 1e-9)

    motion_score = (
        55.0 * progress
        + 35.0 * time_eff
        + 10.0 * vx_start_norm
        + 6.0 * vx_tail_norm
    )

    stability_score = (
        90.0 * float(not out_lane)
        - 40.0 * (max_abs_y / lateral_bound)
        - 25.0 * (mean_abs_y / lateral_bound)
    )

    energy_proxy = (
        0.40 * vx_start_norm ** 2
        + 0.35 * vx_tail_norm ** 2
        + 0.10 * h_dev ** 2
        + 0.15 * c_dev ** 2
    )
    energy_score = -30.0 * energy_proxy

    base_score = (
        100.0 * float(success)
        - 120.0 * float(out_lane)
        - 100.0 * float(startup_failed)
        - 40.0 * float((not goal) and (not startup_failed))
    )

    return (
        base_score
        + beta[0] * motion_score
        + beta[1] * stability_score
        + beta[2] * energy_score
    )


def aggregate_candidates(rows, beta, goal_x, lateral_bound, timeout_s):
    groups = {}
    for r in rows:
        label = r["label"]
        groups.setdefault(label, {
            "rows": [],
            "schedule": r["schedule"],
            "returns": [],
        })
        groups[label]["rows"].append(r)
        groups[label]["returns"].append(
            objective_return(r, beta, goal_x, lateral_bound, timeout_s)
        )

    out = []
    for label, g in groups.items():
        rs = g["rows"]
        success_rate = sum(parse_bool(r.get("success", False)) for r in rs) / len(rs)
        raw_score = sum(ffloat(r, "score", 0.0) for r in rs) / len(rs)
        beta_return = sum(g["returns"]) / len(g["returns"])
        time_vals = [
            float(r["time"]) for r in rs
            if r.get("time", "") not in ["", "NA", "None", None]
        ]
        time_mean = sum(time_vals) / len(time_vals) if time_vals else float("nan")
        max_abs_y = sum(ffloat(r, "max_abs_y", 0.0) for r in rs) / len(rs)
        mean_abs_y = sum(ffloat(r, "mean_abs_y", 0.0) for r in rs) / len(rs)
        a = parse_schedule(g["schedule"])

        out.append({
            "label": label,
            "schedule": g["schedule"],
            "n": len(rs),
            "success_rate": success_rate,
            "raw_score": raw_score,
            "beta_return": beta_return,
            "time_mean": time_mean,
            "max_abs_y_mean": max_abs_y,
            "mean_abs_y_mean": mean_abs_y,
            **a,
        })

    out.sort(key=lambda x: x["beta_return"], reverse=True)
    return out


def beta_features(beta):
    bm, bs, be = beta
    return np.array([
        1.0,
        bm,
        bs,
        be,
        bm - bs,
        bm * bs,
        bm * be,
        bs * be,
        bm ** 2,
        bs ** 2,
        be ** 2,
    ], dtype=np.float64)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def cmd_select(args):
    rows = load_rollouts(args.rollouts)
    ranked = aggregate_candidates(
        rows, args.beta, args.goal_x, args.lateral_bound, args.timeout_s
    )

    best = ranked[0]
    print("[TRACER:D3c] beta =", args.beta)
    print("[TRACER:D3c] selected =", best["label"])
    print("[TRACER:D3c] schedule =", best["schedule"])

    print("\nrank,label,beta_return,success_rate,raw_score,time,max_abs_y,mean_abs_y")
    for i, r in enumerate(ranked, 1):
        t = "NA" if math.isnan(r["time_mean"]) else f"{r['time_mean']:.3f}"
        print(
            f"{i},{r['label']},{r['beta_return']:.3f},{r['success_rate']:.3f},"
            f"{r['raw_score']:.3f},{t},{r['max_abs_y_mean']:.3f},{r['mean_abs_y_mean']:.3f}"
        )

    if args.out_bank:
        label = args.line_label or f"d3c_{best['label']}"
        Path(args.out_bank).parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with open(args.out_bank, mode) as f:
            f.write(f"{label} {best['schedule']}\n")
        print(f"[TRACER:D3c] wrote bank line: {args.out_bank}")


def cmd_make_teacher(args):
    rows = load_rollouts(args.rollouts)
    N = int(round(1.0 / args.step))

    teacher_rows = []
    action_meta = {}

    for im in range(N + 1):
        for is_ in range(N - im + 1):
            ie = N - im - is_
            beta = [im / N, is_ / N, ie / N]

            ranked = aggregate_candidates(
                rows, beta, args.goal_x, args.lateral_bound, args.timeout_s
            )
            best = ranked[0]
            action_meta[best["label"]] = {
                "schedule": best["schedule"],
                "vx_start": best["vx_start"],
                "vx_tail": best["vx_tail"],
                "body_h": best["body_h"],
                "clearance": best["clearance"],
            }

            teacher_rows.append({
                "beta_motion": beta[0],
                "beta_stability": beta[1],
                "beta_energy": beta[2],
                "label": best["label"],
                "beta_return": best["beta_return"],
                "success_rate": best["success_rate"],
                "raw_score": best["raw_score"],
                "schedule": best["schedule"],
                "vx_start": best["vx_start"],
                "vx_tail": best["vx_tail"],
                "body_h": best["body_h"],
                "clearance": best["clearance"],
            })

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        fieldnames = list(teacher_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(teacher_rows)

    counts = {}
    for r in teacher_rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    meta = {
        "rollouts": args.rollouts,
        "step": args.step,
        "num_rows": len(teacher_rows),
        "label_counts": counts,
        "action_meta": action_meta,
    }

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(meta, indent=2))

    print(f"[TRACER:D3c] wrote teacher CSV: {args.out_csv}")
    print("[TRACER:D3c] label counts:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


def cmd_train(args):
    rows = []
    with open(args.teacher_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])

    label_to_idx = {l: i for i, l in enumerate(labels)}

    X = np.stack([
        beta_features([
            float(r["beta_motion"]),
            float(r["beta_stability"]),
            float(r["beta_energy"]),
        ])
        for r in rows
    ])
    y = np.array([label_to_idx[r["label"]] for r in rows], dtype=np.int64)

    K = len(labels)
    D = X.shape[1]
    W = np.zeros((D, K), dtype=np.float64)

    for epoch in range(args.epochs):
        logits = X @ W
        P = softmax(logits)

        Y = np.zeros_like(P)
        Y[np.arange(len(y)), y] = 1.0

        grad = X.T @ (P - Y) / len(y)
        grad += args.l2 * W
        W -= args.lr * grad

        if epoch % 200 == 0 or epoch == args.epochs - 1:
            pred = np.argmax(P, axis=1)
            acc = np.mean(pred == y)
            loss = -np.mean(np.log(P[np.arange(len(y)), y] + 1e-12))
            print(f"epoch={epoch:04d} loss={loss:.4f} acc={acc:.3f}")

    pred = np.argmax(softmax(X @ W), axis=1)
    acc = float(np.mean(pred == y))

    action_meta = {}
    for r in rows:
        label = r["label"]
        action_meta[label] = {
            "schedule": r["schedule"],
            "vx_start": float(r["vx_start"]),
            "vx_tail": float(r["vx_tail"]),
            "body_h": float(r["body_h"]),
            "clearance": float(r["clearance"]),
        }

    model = {
        "version": "tracer_phase_d3c_discrete_policy_v0",
        "labels": labels,
        "weights": W.tolist(),
        "train_accuracy": acc,
        "feature_names": [
            "bias",
            "beta_motion",
            "beta_stability",
            "beta_energy",
            "beta_motion_minus_stability",
            "beta_motion_times_stability",
            "beta_motion_times_energy",
            "beta_stability_times_energy",
            "beta_motion_sq",
            "beta_stability_sq",
            "beta_energy_sq",
        ],
        "action_meta": action_meta,
    }

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_model).write_text(json.dumps(model, indent=2))

    print(f"[TRACER:D3c] wrote model: {args.out_model}")
    print(f"[TRACER:D3c] train_accuracy={acc:.3f}")
    print("[TRACER:D3c] labels:", labels)


def cmd_predict(args):
    model = json.loads(Path(args.model).read_text())
    labels = model["labels"]
    W = np.array(model["weights"], dtype=np.float64)
    x = beta_features(args.beta)[None, :]
    p = softmax(x @ W)[0]

    order = np.argsort(-p)
    best_idx = int(order[0])
    best_label = labels[best_idx]
    meta = model["action_meta"][best_label]

    print("[TRACER:D3c] beta =", args.beta)
    print("[TRACER:D3c] selected =", best_label)
    print("[TRACER:D3c] prob =", float(p[best_idx]))
    print("[TRACER:D3c] schedule =", meta["schedule"])
    print("\nrank,label,prob")
    for i, idx in enumerate(order, 1):
        print(f"{i},{labels[int(idx)]},{float(p[int(idx)]):.4f}")

    if args.out_bank:
        line_label = args.line_label or f"d3c_{best_label}"
        Path(args.out_bank).parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with open(args.out_bank, mode) as f:
            f.write(f"{line_label} {meta['schedule']}\n")
        print(f"[TRACER:D3c] wrote bank line: {args.out_bank}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("select")
    p.add_argument("--rollouts", required=True)
    p.add_argument("--beta", nargs=3, type=float, required=True)
    p.add_argument("--goal-x", type=float, default=8.0)
    p.add_argument("--lateral-bound", type=float, default=2.0)
    p.add_argument("--timeout-s", type=float, default=460.0)
    p.add_argument("--out-bank", default="")
    p.add_argument("--line-label", default="")
    p.add_argument("--append", action="store_true")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("make-teacher")
    p.add_argument("--rollouts", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-json", default="")
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--goal-x", type=float, default=8.0)
    p.add_argument("--lateral-bound", type=float, default=2.0)
    p.add_argument("--timeout-s", type=float, default=460.0)
    p.set_defaults(func=cmd_make_teacher)

    p = sub.add_parser("train")
    p.add_argument("--teacher-csv", required=True)
    p.add_argument("--out-model", required=True)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--l2", type=float, default=1e-4)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--beta", nargs=3, type=float, required=True)
    p.add_argument("--out-bank", default="")
    p.add_argument("--line-label", default="")
    p.add_argument("--append", action="store_true")
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
