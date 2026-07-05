#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateScorer(nn.Module):
    def __init__(self, input_dim, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def read_jsonl(p):
    rows = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


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


def load_task_beta_targets(path):
    rep = load_json(path)
    targets = {}
    meta = {}
    for r in rep["summary"]:
        k = r["world_action"]
        targets[k] = float(r["score_task_beta"])
        meta[k] = r
    return targets, meta


def collect_feature_names(rows):
    names = set()
    for r in rows:
        feat = r.get("features") or {}
        for k in feat.keys():
            if k.startswith("metric:") or k.startswith("reward:"):
                continue
            names.add(k)
    return sorted(names)


def build_x(row, feature_names, target_meta):
    feat = row.get("features") or {}
    w = row.get("world_name", "")
    a = row.get("action_name", "")
    key = f"{w}::{a}"
    m = target_meta.get(key, {})

    vals = []
    for k in feature_names:
        if k.startswith("world_onehot:") and k not in feat:
            vals.append(1.0 if k.split("world_onehot:", 1)[1] == w else 0.0)
        else:
            vals.append(ff(feat.get(k), 0.0))

    # Deployable beta inputs.
    # These can be produced by Objective Selector beta model at runtime.
    vals += [
        ff(m.get("mean_beta_v")),
        ff(m.get("mean_beta_s")),
        ff(m.get("mean_beta_e")),
    ]

    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="data/phase_b_training_pipeline_v5_merged_balanced/phase_b_rollout_index_v1.jsonl")
    ap.add_argument("--task-beta-report", default="reports/phase_b_task_beta_reward_v0.json")
    ap.add_argument("--out-dir", default="artifacts/phase_b_beta_aware_candidate_policy_v7_deployable")
    ap.add_argument("--report-json", default="reports/phase_b_beta_aware_candidate_policy_v7_deployable.json")
    ap.add_argument("--report-md", default="reports/phase_b_beta_aware_candidate_policy_v7_deployable.md")
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--pair-loss-weight", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=107)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.rollouts)
    targets, target_meta = load_task_beta_targets(args.task_beta_report)

    usable = []
    for r in rows:
        key = f"{r.get('world_name')}::{r.get('action_name')}"
        if key in targets:
            usable.append(r)

    feature_names = collect_feature_names(usable)
    extra_names = ["beta_v", "beta_s", "beta_e"]
    input_names = feature_names + extra_names

    X = torch.tensor([build_x(r, feature_names, target_meta) for r in usable], dtype=torch.float32)
    y = torch.tensor([targets[f"{r.get('world_name')}::{r.get('action_name')}"] for r in usable], dtype=torch.float32)

    x_mean = X.mean(dim=0)
    x_std = X.std(dim=0).clamp_min(1e-6)
    y_mean = y.mean()
    y_std = y.std().clamp_min(1e-6)

    Xn = (X - x_mean) / x_std
    yn = (y - y_mean) / y_std

    idx = list(range(len(usable)))
    random.shuffle(idx)
    n_val = max(1, int(0.2 * len(idx)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    model = CandidateScorer(input_dim=Xn.shape[1], hidden=96)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    by_world = defaultdict(list)
    for i, r in enumerate(usable):
        by_world[r.get("world_name")].append(i)

    pairs = []
    for _, ids in by_world.items():
        for i in ids:
            for j in ids:
                if y[i] > y[j] + 1e-6:
                    pairs.append((i, j))

    best = {"epoch": -1, "val_mse": 1e9, "state": None}
    hist = []

    for ep in range(1, args.epochs + 1):
        model.train()
        order = train_idx[:]
        random.shuffle(order)

        pred = model(Xn[order])
        mse = F.mse_loss(pred, yn[order])

        pair_loss = torch.tensor(0.0)
        if pairs:
            sample_pairs = random.sample(pairs, min(len(pairs), 512))
            pi = torch.tensor([p[0] for p in sample_pairs], dtype=torch.long)
            pj = torch.tensor([p[1] for p in sample_pairs], dtype=torch.long)
            si = model(Xn[pi])
            sj = model(Xn[pj])
            pair_loss = F.softplus(-(si - sj)).mean()

        loss = mse + args.pair_loss_weight * pair_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            train_mse = F.mse_loss(model(Xn[train_idx]), yn[train_idx]).item()
            val_mse = F.mse_loss(model(Xn[val_idx]), yn[val_idx]).item()

            correct = 0
            total = 0
            for i, j in pairs:
                if float(model(Xn[i].unsqueeze(0))[0]) > float(model(Xn[j].unsqueeze(0))[0]):
                    correct += 1
                total += 1
            pair_acc = correct / max(total, 1)

        rec = {
            "epoch": ep,
            "loss": float(loss.item()),
            "train_mse": train_mse,
            "val_mse": val_mse,
            "pair_acc": pair_acc,
        }
        hist.append(rec)

        if val_mse < best["val_mse"]:
            best = {
                "epoch": ep,
                "val_mse": val_mse,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

        if ep == 1 or ep % 100 == 0 or ep == args.epochs:
            print(json.dumps(rec, sort_keys=True))

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    model.eval()
    with torch.no_grad():
        scores = model(Xn) * y_std + y_mean

    rows_out = []
    for r, s, t in zip(usable, scores.tolist(), y.tolist()):
        key = f"{r.get('world_name')}::{r.get('action_name')}"
        rows_out.append({
            "world_name": r.get("world_name"),
            "action_name": r.get("action_name"),
            "world_action": key,
            "pred_score": float(s),
            "target_score_task_beta": float(t),
            "target_meta": target_meta.get(key, {}),
        })

    groups = defaultdict(list)
    for r in rows_out:
        groups[r["world_action"]].append(r)

    group_summary = []
    for k, arr in sorted(groups.items()):
        group_summary.append({
            "world_action": k,
            "n": len(arr),
            "mean_pred_score": sum(x["pred_score"] for x in arr) / len(arr),
            "target_score_task_beta": arr[0]["target_score_task_beta"],
            "target_proxy": ff(arr[0]["target_meta"].get("mean_proxy_reward")),
            "target_reach": ff(arr[0]["target_meta"].get("reach_rate")),
            "target_final": ff(arr[0]["target_meta"].get("mean_final_dist")),
            "target_drift": ff(arr[0]["target_meta"].get("mean_drift")),
            "beta_v": ff(arr[0]["target_meta"].get("mean_beta_v")),
            "beta_s": ff(arr[0]["target_meta"].get("mean_beta_s")),
            "beta_e": ff(arr[0]["target_meta"].get("mean_beta_e")),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "schema": "phase_b_beta_aware_candidate_policy_v7_deployable",
        "model_state_dict": model.state_dict(),
        "input_dim": Xn.shape[1],
        "input_names": input_names,
        "input_mean": x_mean.tolist(),
        "input_std": x_std.tolist(),
        "target_mean": float(y_mean.item()),
        "target_std": float(y_std.item()),
        "feature_names_base": feature_names,
        "extra_names": extra_names,
        "task_beta_report": args.task_beta_report,
        "best_epoch": best["epoch"],
        "best_val_mse": best["val_mse"],
        "note": "Deployable beta-aware candidate policy. Inputs use context/theta features and beta only. No group outcome features.",
    }
    torch.save(ckpt, out_dir / "model.pt")

    report = {
        "schema": "phase_b_beta_aware_candidate_policy_v7_deployable_report",
        "rollouts": args.rollouts,
        "task_beta_report": args.task_beta_report,
        "out_dir": str(out_dir),
        "num_rows": len(usable),
        "input_dim": Xn.shape[1],
        "best_epoch": best["epoch"],
        "best_val_mse": best["val_mse"],
        "history_tail": hist[-20:],
        "group_summary": group_summary,
    }
    save_json(out_dir / "train_report.json", report)
    save_json(Path(args.report_json), report)

    lines = []
    lines.append("# Phase-B β-aware Candidate Policy v7 Deployable")
    lines.append("")
    lines.append(f"- Rows: `{len(usable)}`")
    lines.append(f"- Input dim: `{Xn.shape[1]}`")
    lines.append(f"- Best epoch: `{best['epoch']}`")
    lines.append(f"- Best val MSE: `{best['val_mse']:.6f}`")
    lines.append("")
    lines.append("## Group summary")
    lines.append("")
    lines.append("| world::action | n | pred | target_task_beta | proxy | reach | final | drift | beta_v | beta_s | beta_e |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in sorted(group_summary, key=lambda x: (x["world_action"].split("::")[0], -x["mean_pred_score"])):
        lines.append(
            f"| {g['world_action']} | {g['n']} | {g['mean_pred_score']:.3f} | "
            f"{g['target_score_task_beta']:.3f} | {g['target_proxy']:.3f} | "
            f"{g['target_reach']:.3f} | {g['target_final']:.3f} | {g['target_drift']:.3f} | "
            f"{g['beta_v']:.3f} | {g['beta_s']:.3f} | {g['beta_e']:.3f} |"
        )

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
