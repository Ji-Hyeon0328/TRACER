#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

def ffloat(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def bbool(x):
    return str(x).lower() in ("1", "true", "yes")

def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

FEATURES = [
    "bias",
    "reset_y",
    "abs_reset_y",
    "learned_vx",
    "learned_yaw",
    "learned_clearance",
    "abs_reset_y_x_learned_vx",
    "abs_reset_y_x_learned_yaw",
    "abs_reset_y_x_learned_clearance",
    "reset_y_x_learned_vx",
    "reset_y_x_learned_yaw",
    "reset_y_x_learned_clearance",
]

def feat(r):
    reset_y = ffloat(r.get("reset_y"))
    abs_reset_y = abs(reset_y)
    lvx = 1.0 if bbool(r.get("learned_vx")) else 0.0
    lyaw = 1.0 if bbool(r.get("learned_yaw")) else 0.0
    lclr = 1.0 if bbool(r.get("learned_clearance")) else 0.0

    return [
        1.0,
        reset_y,
        abs_reset_y,
        lvx,
        lyaw,
        lclr,
        abs_reset_y * lvx,
        abs_reset_y * lyaw,
        abs_reset_y * lclr,
        reset_y * lvx,
        reset_y * lyaw,
        reset_y * lclr,
    ]

def dot(w, x):
    return sum(a*b for a, b in zip(w, x))

def dims(r):
    return f"{r.get('learned_vx')}/{r.get('learned_yaw')}/{r.get('learned_clearance')}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", required=True)
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    args = ap.parse_args()

    metric_rows = list(csv.DictReader(open(args.metrics_csv)))
    pair_rows = list(csv.DictReader(open(args.pairs_csv)))

    if not pair_rows:
        raise SystemExit("No preference pairs")

    xs = [feat(r) for r in metric_rows]
    w = [0.0 for _ in FEATURES]

    pair_indices = []
    for p in pair_rows:
        i = int(p["preferred_idx"])
        j = int(p["rejected_idx"])
        pair_indices.append((i, j, p))

    history = []
    n = len(pair_indices)

    for ep in range(args.epochs):
        grad = [0.0 for _ in w]
        loss = 0.0
        correct = 0

        for i, j, _p in pair_indices:
            xd = [a-b for a, b in zip(xs[i], xs[j])]
            z = dot(w, xd)
            prob = sigmoid(z)

            loss += -math.log(max(prob, 1e-12))
            correct += 1 if z > 0 else 0

            # gradient of -log(sigmoid(z)) = prob - 1
            coeff = prob - 1.0
            for k in range(len(w)):
                grad[k] += coeff * xd[k]

        for k in range(len(w)):
            loss += 0.5 * args.l2 * w[k] * w[k]
            grad[k] += args.l2 * w[k]

        for k in range(len(w)):
            w[k] -= args.lr * grad[k] / max(1, n)

        if ep in (0, 1, 2, 5, 10, 50, 100, 500, 1000, 2000, args.epochs - 1):
            history.append({
                "epoch": ep,
                "loss": loss / max(1, n),
                "pair_acc": correct / max(1, n),
            })

    scored = []
    for idx, r in enumerate(metric_rows):
        s = dot(w, xs[idx])
        rr = dict(r)
        rr["_idx"] = idx
        rr["_score"] = s
        rr["_dims"] = dims(r)
        scored.append(rr)

    correct = 0
    pair_examples = []
    for i, j, p in pair_indices:
        si = scored[i]["_score"]
        sj = scored[j]["_score"]
        ok = si > sj
        correct += 1 if ok else 0
        pair_examples.append({
            "pref_type": p.get("pref_type"),
            "reset_y": p.get("reset_y"),
            "preferred_idx": i,
            "rejected_idx": j,
            "preferred_dims": p.get("preferred_dims"),
            "rejected_dims": p.get("rejected_dims"),
            "preferred_score": si,
            "rejected_score": sj,
            "model_margin": si - sj,
            "label_score_diff": ffloat(p.get("score_diff")),
            "ok": ok,
        })

    pair_acc = correct / max(1, len(pair_indices))

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    model = {
        "model_name": "phase_e3_preference_fallback_scorer_v0",
        "kind": "pairwise_logistic_bootstrap",
        "note": "This is not true IRL. It is a weak preference-supervised fallback/deployment scorer trained from E2 filtered pairs.",
        "features": FEATURES,
        "weights": {name: value for name, value in zip(FEATURES, w)},
        "metrics_csv": args.metrics_csv,
        "pairs_csv": args.pairs_csv,
        "num_metric_rows": len(metric_rows),
        "num_pairs": len(pair_rows),
        "epochs": args.epochs,
        "lr": args.lr,
        "l2": args.l2,
        "pair_accuracy": pair_acc,
        "history": history,
    }
    out_json.write_text(json.dumps(model, indent=2))

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with open(out_md, "w") as f:
        f.write("# TRACER Phase-E3 Preference Fallback Scorer v0\n\n")
        f.write("This is a pairwise-logistic bootstrap scorer trained from E2 filtered preference pairs. It is not a true IRL Objective Selector; it is a weak preference-supervised fallback/deployment scorer.\n\n")
        f.write(f"- metrics_csv: `{args.metrics_csv}`\n")
        f.write(f"- pairs_csv: `{args.pairs_csv}`\n")
        f.write(f"- out_json: `{out_json}`\n")
        f.write(f"- num_pairs: `{len(pair_rows)}`\n")
        f.write(f"- pair_accuracy: `{pair_acc:.3f}`\n")
        f.write(f"- epochs: `{args.epochs}`\n")
        f.write(f"- lr: `{args.lr}`\n")
        f.write(f"- l2: `{args.l2}`\n\n")

        f.write("## Weights\n\n")
        f.write("| feature | weight |\n")
        f.write("|---|---:|\n")
        for name, value in zip(FEATURES, w):
            f.write(f"| {name} | {value:.6f} |\n")

        f.write("\n## Training history\n\n")
        f.write("| epoch | loss | pair_acc |\n")
        f.write("|---:|---:|---:|\n")
        for h in history:
            f.write(f"| {h['epoch']} | {h['loss']:.6f} | {h['pair_acc']:.3f} |\n")

        f.write("\n## Rollout scores\n\n")
        f.write("| idx | reset_y | dims | score | final_y | path_max_abs_y | moving_accept |\n")
        f.write("|---:|---:|---|---:|---:|---:|---:|\n")
        for r in sorted(scored, key=lambda x: x["_score"], reverse=True):
            f.write(
                f"| {r['_idx']} | {float(r['reset_y']):+.2f} | {r['_dims']} | "
                f"{r['_score']:.4f} | {float(r['final_y']):+.3f} | "
                f"{float(r['path_max_abs_y']):.3f} | {float(r['moving_accept_rate']):.3f} |\n"
            )

        f.write("\n## Top pair predictions\n\n")
        f.write("| pref_type | reset_y | preferred | rejected | model_margin | label_diff | ok |\n")
        f.write("|---|---:|---|---|---:|---:|---|\n")
        for p in sorted(pair_examples, key=lambda x: abs(x["model_margin"]), reverse=True)[:30]:
            f.write(
                f"| {p['pref_type']} | {float(p['reset_y']):+.2f} | "
                f"{p['preferred_dims']} | {p['rejected_dims']} | "
                f"{p['model_margin']:.4f} | {p['label_score_diff']:.4f} | {p['ok']} |\n"
            )

    print(f"[TRACER] wrote {out_json}")
    print(f"[TRACER] wrote {out_md}")
    print(f"[TRACER] pair_accuracy={pair_acc:.3f}")

if __name__ == "__main__":
    main()
