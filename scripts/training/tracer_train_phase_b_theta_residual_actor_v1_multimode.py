#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


THETA_KEYS = [
    "vx_far",
    "vx_near",
    "goal_slow_distance",
    "goal_stop_distance",
    "body_height",
    "swing_clearance",
]

WORLD_KEYS = [
    "earth",
    "stairs_single",
    "tracer_sponge_firm_flat",
]

TARGET_MODES = [
    "balanced",
    "reach",
    "stability",
]

ANCHOR_SCALE = {
    "vx_far": 0.30,
    "vx_near": 0.10,
    "goal_slow_distance": 0.85,
    "goal_stop_distance": 0.30,
    "body_height": 0.36,
    "swing_clearance": 0.12,
}


class ResidualActorV1(nn.Module):
    def __init__(self, input_dim, output_dim=6, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def read_jsonl(p):
    rows = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def feature_vec(r):
    x = []

    world = r.get("world", "")
    for w in WORLD_KEYS:
        x.append(1.0 if world == w else 0.0)

    mode = r.get("target_mode", "")
    for m in TARGET_MODES:
        x.append(1.0 if mode == m else 0.0)

    x.extend([
        float(r.get("beta_v", 0.0)),
        float(r.get("beta_s", 0.0)),
        float(r.get("beta_e", 0.0)),
    ])

    for k in THETA_KEYS:
        x.append(float(r[f"anchor_{k}"]) / ANCHOR_SCALE[k])

    x.append(float(r.get("anchor_score_task_beta", 0.0)))

    return x


def target_vec(r):
    return [float(r[f"delta_norm_{k}"]) for k in THETA_KEYS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/phase_b_theta_residual_actor_v1_multimode/phase_b_theta_residual_actor_seed_v1_multimode.jsonl")
    ap.add_argument("--out-dir", default="artifacts/phase_b_theta_residual_actor_v1_multimode")
    ap.add_argument("--report-json", default="reports/phase_b_theta_residual_actor_v1_multimode.json")
    ap.add_argument("--report-md", default="reports/phase_b_theta_residual_actor_v1_multimode.md")
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.dataset)
    random.shuffle(rows)

    n = len(rows)
    n_val = max(1, int(0.20 * n))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    X_train = torch.tensor([feature_vec(r) for r in train_rows], dtype=torch.float32)
    Y_train = torch.tensor([target_vec(r) for r in train_rows], dtype=torch.float32)
    W_train = torch.tensor([float(r.get("train_weight", 1.0)) for r in train_rows], dtype=torch.float32).view(-1, 1)

    X_val = torch.tensor([feature_vec(r) for r in val_rows], dtype=torch.float32)
    Y_val = torch.tensor([target_vec(r) for r in val_rows], dtype=torch.float32)

    model = ResidualActorV1(input_dim=X_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = {"epoch": None, "val_mse": 1e9, "state": None}

    for ep in range(1, args.epochs + 1):
        model.train()
        pred = model(X_train)
        loss_raw = F.mse_loss(pred, Y_train, reduction="none").mean(dim=1, keepdim=True)
        loss = (loss_raw * W_train).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep == 1 or ep % 100 == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                train_mse = F.mse_loss(model(X_train), Y_train).item()
                val_mse = F.mse_loss(model(X_val), Y_val).item()

            if val_mse < best["val_mse"]:
                best = {
                    "epoch": ep,
                    "val_mse": val_mse,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }

            print(json.dumps({
                "epoch": ep,
                "loss": float(loss.item()),
                "train_mse": train_mse,
                "val_mse": val_mse,
            }))

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    group = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for r in rows:
            x = torch.tensor([feature_vec(r)], dtype=torch.float32)
            p = model(x)[0].tolist()
            y = target_vec(r)
            key = f"{r['world']}::{r['target_mode']}::{r['anchor_action']} -> {r['target_action']}"
            group[key].append((p, y))

    group_summary = []
    for key, vals in sorted(group.items()):
        pred_mean = [sum(v[0][i] for v in vals) / len(vals) for i in range(len(THETA_KEYS))]
        targ_mean = [sum(v[1][i] for v in vals) / len(vals) for i in range(len(THETA_KEYS))]
        mse = sum((pred_mean[i] - targ_mean[i]) ** 2 for i in range(len(THETA_KEYS))) / len(THETA_KEYS)
        group_summary.append({
            "group": key,
            "n": len(vals),
            "mse_mean": mse,
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "schema": "phase_b_theta_residual_actor_v1_multimode",
        "model_state_dict": model.state_dict(),
        "input_dim": X_train.shape[1],
        "output_dim": len(THETA_KEYS),
        "theta_keys": THETA_KEYS,
        "world_keys": WORLD_KEYS,
        "target_modes": TARGET_MODES,
        "anchor_scale": ANCHOR_SCALE,
        "feature_schema": [
            "world_onehot",
            "target_mode_onehot",
            "beta_v,beta_s,beta_e",
            "anchor_theta_scaled",
            "anchor_score_task_beta",
        ],
    }

    torch.save(ckpt, out_dir / "model.pt")

    report = {
        "schema": "phase_b_theta_residual_actor_train_report_v1_multimode",
        "dataset": args.dataset,
        "rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "input_dim": X_train.shape[1],
        "best_epoch": best["epoch"],
        "best_val_mse": best["val_mse"],
        "group_summary": group_summary,
    }

    with open(out_dir / "train_report.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    lines = []
    lines.append("# Phase-B θ Residual Actor v1 Multimode")
    lines.append("")
    lines.append(f"- Rows: `{len(rows)}`")
    lines.append(f"- Train rows: `{len(train_rows)}`")
    lines.append(f"- Val rows: `{len(val_rows)}`")
    lines.append(f"- Input dim: `{X_train.shape[1]}`")
    lines.append(f"- Best epoch: `{best['epoch']}`")
    lines.append(f"- Best val MSE: `{best['val_mse']:.6f}`")
    lines.append("")
    lines.append("## Group summary")
    lines.append("")
    lines.append("| group | n | mse_mean |")
    lines.append("|---|---:|---:|")
    for g in group_summary:
        lines.append(f"| {g['group']} | {g['n']} | {g['mse_mean']:.6f} |")

    md = "\n".join(lines) + "\n"

    with open(out_dir / "train_report.md", "w") as f:
        f.write(md)
    with open(args.report_md, "w") as f:
        f.write(md)

    print(md)
    print("[wrote]", out_dir / "model.pt")


if __name__ == "__main__":
    main()
