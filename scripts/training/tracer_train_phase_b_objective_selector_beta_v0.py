#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def read_jsonl(p):
    rows = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def ff(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def get_reward_components(r):
    c = r.get("reward_components") or {}
    return {
        "R_v": ff(c.get("R_v")),
        "R_s": ff(c.get("R_s")),
        "R_e": ff(c.get("R_e")),
        "final_rel_dist": ff(c.get("final_rel_dist")),
        "min_rel_dist": ff(c.get("min_rel_dist")),
        "post_reach_drift": ff(c.get("post_reach_drift")),
        "reached": 1.0 if c.get("reached_stop_distance") else 0.0,
        "hold_reward": ff(c.get("hold_reward")),
        "reach_reward": ff(c.get("reach_reward")),
    }


def beta_label_from_context_and_outcome(r):
    """
    Bootstrap pseudo-label for β=[β_v,β_s,β_e].
    This is not final IRL; it initializes Objective Selector with explicit reward weights.
    """
    w = r.get("world_name") or r.get("world") or ""
    rc = get_reward_components(r)
    labels = r.get("ram_labels") or {}
    risk = ff(labels.get("future_risk_proxy"), default=0.5)

    if w == "earth":
        beta = [0.58, 0.24, 0.18]
    elif w == "stairs_single":
        beta = [0.46, 0.40, 0.14]
    elif "sponge" in w:
        beta = [0.28, 0.58, 0.14]
    else:
        beta = [0.45, 0.40, 0.15]

    drift = rc["post_reach_drift"]
    final_d = rc["final_rel_dist"]
    reached = rc["reached"]

    # If it reaches but drifts away, increase stability weight.
    if reached > 0.5 and (drift > 0.75 or final_d > 1.0):
        beta[0] -= 0.16
        beta[1] += 0.20
        beta[2] -= 0.04

    # If it does not reach and remains far, progress should matter more,
    # but soft terrain should still not over-prioritize speed.
    if reached < 0.5 and final_d > 0.8:
        if "sponge" in w:
            beta[0] += 0.04
            beta[1] += 0.04
            beta[2] -= 0.08
        else:
            beta[0] += 0.12
            beta[1] -= 0.04
            beta[2] -= 0.08

    # High predicted/outcome risk should bias toward stability.
    if risk > 0.65:
        beta[0] -= 0.08
        beta[1] += 0.12
        beta[2] -= 0.04

    # Stable reach can afford more velocity/energy balance.
    if reached > 0.5 and drift < 0.25 and final_d < 0.5:
        beta[0] += 0.04
        beta[1] -= 0.02
        beta[2] += 0.02

    beta = [max(0.05, x) for x in beta]
    s = sum(beta)
    return [x / s for x in beta]


class BetaSelectorNet(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
        )
        self.beta_head = nn.Linear(32, 3)

    def forward(self, x):
        h = self.trunk(x)
        return self.beta_head(h)


def collect_feature_names(rows):
    names = set()
    for r in rows:
        feat = r.get("features") or {}
        for k in feat.keys():
            if k.startswith("metric:") or k.startswith("reward:"):
                continue
            if k.startswith("theta:"):
                continue
            names.add(k)

    # Fallback if features are sparse.
    if not names:
        worlds = sorted(set(r.get("world_name", "") for r in rows))
        return [f"world_onehot:{w}" for w in worlds]

    return sorted(names)


def build_feature_vec(r, names):
    feat = r.get("features") or {}
    w = r.get("world_name", "")
    vals = []
    for k in names:
        if k.startswith("world_onehot:") and k not in feat:
            vals.append(1.0 if k.split("world_onehot:", 1)[1] == w else 0.0)
        else:
            vals.append(ff(feat.get(k), 0.0))
    return torch.tensor(vals, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="data/phase_b_training_pipeline_v5_merged_balanced/phase_b_rollout_index_v1.jsonl")
    ap.add_argument("--out-dir", default="artifacts/phase_b_objective_selector_beta_v0")
    ap.add_argument("--report-json", default="reports/phase_b_objective_selector_beta_v0.json")
    ap.add_argument("--report-md", default="reports/phase_b_objective_selector_beta_v0.md")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=91)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.rollouts)
    feature_names = collect_feature_names(rows)

    X = torch.stack([build_feature_vec(r, feature_names) for r in rows], dim=0)
    Y = torch.tensor([beta_label_from_context_and_outcome(r) for r in rows], dtype=torch.float32)

    x_mean = X.mean(dim=0)
    x_std = X.std(dim=0).clamp_min(1e-6)
    Xn = (X - x_mean) / x_std

    idx = list(range(len(rows)))
    random.shuffle(idx)
    n_val = max(1, int(0.2 * len(idx)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    model = BetaSelectorNet(input_dim=len(feature_names), hidden=64)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = {"epoch": -1, "val_kl": 1e9, "state": None}
    hist = []

    for ep in range(1, args.epochs + 1):
        model.train()
        order = train_idx[:]
        random.shuffle(order)
        losses = []

        for i in order:
            logits = model(Xn[i].unsqueeze(0))[0]
            target = Y[i]
            logp = F.log_softmax(logits, dim=0)
            loss = F.kl_div(logp, target, reduction="sum")

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            train_pred = F.log_softmax(model(Xn[train_idx]), dim=1)
            val_pred = F.log_softmax(model(Xn[val_idx]), dim=1)
            train_kl = F.kl_div(train_pred, Y[train_idx], reduction="batchmean").item()
            val_kl = F.kl_div(val_pred, Y[val_idx], reduction="batchmean").item()

        rec = {
            "epoch": ep,
            "loss": sum(losses) / max(len(losses), 1),
            "train_kl": train_kl,
            "val_kl": val_kl,
        }
        hist.append(rec)

        if val_kl < best["val_kl"]:
            best = {
                "epoch": ep,
                "val_kl": val_kl,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

        if ep == 1 or ep % 100 == 0 or ep == args.epochs:
            print(json.dumps(rec, sort_keys=True))

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # Aggregate prediction by world.
    model.eval()
    by_world = {}
    by_world_action = {}
    with torch.no_grad():
        probs = F.softmax(model(Xn), dim=1).cpu().tolist()
    for r, b in zip(rows, probs):
        w = r.get("world_name", "")
        a = r.get("action_name", "")
        by_world.setdefault(w, []).append(b)
        by_world_action.setdefault(f"{w}::{a}", []).append(b)

    def avg_beta(arr):
        n = len(arr)
        return [sum(x[j] for x in arr) / n for j in range(3)]

    world_summary = {k: avg_beta(v) for k, v in sorted(by_world.items())}
    action_summary = {k: avg_beta(v) for k, v in sorted(by_world_action.items())}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "schema": "phase_b_objective_selector_beta_v0",
        "model_state_dict": model.state_dict(),
        "input_dim": len(feature_names),
        "feature_names": feature_names,
        "feature_mean": x_mean.tolist(),
        "feature_std": x_std.tolist(),
        "beta_names": ["beta_v", "beta_s", "beta_e"],
        "best_epoch": best["epoch"],
        "best_val_kl": best["val_kl"],
        "note": "Bootstrap explicit beta selector. Inputs exclude theta/reward/metric features.",
    }
    torch.save(ckpt, out_dir / "model.pt")

    report = {
        "schema": "phase_b_objective_selector_beta_v0_report",
        "rollouts": args.rollouts,
        "out_dir": str(out_dir),
        "num_rows": len(rows),
        "input_dim": len(feature_names),
        "feature_names": feature_names,
        "best_epoch": best["epoch"],
        "best_val_kl": best["val_kl"],
        "world_summary": world_summary,
        "world_action_summary": action_summary,
        "history_tail": hist[-20:],
    }
    save_json(out_dir / "train_report.json", report)
    save_json(Path(args.report_json), report)

    lines = []
    lines.append("# Phase-B Objective Selector β v0")
    lines.append("")
    lines.append(f"- Rollouts: `{len(rows)}`")
    lines.append(f"- Input dim: `{len(feature_names)}`")
    lines.append(f"- Best epoch: `{best['epoch']}`")
    lines.append(f"- Best val KL: `{best['val_kl']:.6f}`")
    lines.append("")
    lines.append("β = [β_v, β_s, β_e]")
    lines.append("")
    lines.append("## By world")
    lines.append("")
    lines.append("| world | beta_v | beta_s | beta_e |")
    lines.append("|---|---:|---:|---:|")
    for w, b in world_summary.items():
        lines.append(f"| {w} | {b[0]:.3f} | {b[1]:.3f} | {b[2]:.3f} |")
    lines.append("")
    lines.append("## By world::action")
    lines.append("")
    lines.append("| world::action | beta_v | beta_s | beta_e |")
    lines.append("|---|---:|---:|---:|")
    for k, b in action_summary.items():
        lines.append(f"| {k} | {b[0]:.3f} | {b[1]:.3f} | {b[2]:.3f} |")

    md = "\n".join(lines) + "\n"
    (out_dir / "train_report.md").write_text(md)
    Path(args.report_md).write_text(md)

    print()
    print(md)
    print("[wrote]", out_dir / "model.pt")
    print("[wrote]", out_dir / "train_report.json")
    print("[wrote]", out_dir / "train_report.md")
    print("[wrote]", args.report_json)
    print("[wrote]", args.report_md)


if __name__ == "__main__":
    main()
