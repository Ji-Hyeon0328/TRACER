#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def build_vec(r, names):
    feat = r.get("features") or {}
    w = r.get("world_name", "")
    vals = []
    for k in names:
        if k.startswith("world_onehot:") and k not in feat:
            vals.append(1.0 if k.split("world_onehot:", 1)[1] == w else 0.0)
        else:
            vals.append(ff(feat.get(k), 0.0))
    return torch.tensor(vals, dtype=torch.float32)


def load_beta(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = BetaSelectorNet(ckpt["input_dim"], hidden=64)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="data/phase_b_training_pipeline_v5_merged_balanced/phase_b_rollout_index_v1.jsonl")
    ap.add_argument("--beta-model", default="artifacts/phase_b_objective_selector_beta_v0/model.pt")
    ap.add_argument("--out-json", default="reports/phase_b_beta_weighted_reward_v0.json")
    ap.add_argument("--out-md", default="reports/phase_b_beta_weighted_reward_v0.md")
    args = ap.parse_args()

    rows = read_jsonl(args.rollouts)
    ckpt, model = load_beta(args.beta_model)

    names = ckpt["feature_names"]
    mean = torch.tensor(ckpt["feature_mean"], dtype=torch.float32)
    std = torch.tensor(ckpt["feature_std"], dtype=torch.float32)

    scored = []
    with torch.no_grad():
        for r in rows:
            x = (build_vec(r, names) - mean) / std
            beta = F.softmax(model(x.unsqueeze(0))[0], dim=0).cpu().tolist()
            c = r.get("reward_components") or {}
            R_v = ff(c.get("R_v"))
            R_s = ff(c.get("R_s"))
            R_e = ff(c.get("R_e"))
            score_beta = beta[0] * R_v + beta[1] * R_s + beta[2] * R_e
            scored.append({
                "world_name": r.get("world_name"),
                "action_name": r.get("action_name"),
                "source_path": r.get("source_path"),
                "beta_v": beta[0],
                "beta_s": beta[1],
                "beta_e": beta[2],
                "R_v": R_v,
                "R_s": R_s,
                "R_e": R_e,
                "score_beta": score_beta,
                "reached": bool(c.get("reached_stop_distance")),
                "final_rel_dist": ff(c.get("final_rel_dist")),
                "post_reach_drift": ff(c.get("post_reach_drift")),
                "tracer_proxy_reward": ff(c.get("tracer_proxy_reward")),
            })

    groups = defaultdict(list)
    for r in scored:
        groups[f"{r['world_name']}::{r['action_name']}"].append(r)

    summary = []
    for k, arr in sorted(groups.items()):
        n = len(arr)
        summary.append({
            "world_action": k,
            "n": n,
            "reach_rate": sum(1.0 if x["reached"] else 0.0 for x in arr) / n,
            "mean_beta_v": sum(x["beta_v"] for x in arr) / n,
            "mean_beta_s": sum(x["beta_s"] for x in arr) / n,
            "mean_beta_e": sum(x["beta_e"] for x in arr) / n,
            "mean_R_v": sum(x["R_v"] for x in arr) / n,
            "mean_R_s": sum(x["R_s"] for x in arr) / n,
            "mean_R_e": sum(x["R_e"] for x in arr) / n,
            "mean_score_beta": sum(x["score_beta"] for x in arr) / n,
            "mean_proxy_reward": sum(x["tracer_proxy_reward"] for x in arr) / n,
            "mean_final_dist": sum(x["final_rel_dist"] for x in arr) / n,
            "mean_drift": sum(x["post_reach_drift"] for x in arr) / n,
        })

    report = {
        "schema": "phase_b_beta_weighted_reward_report_v0",
        "rollouts": args.rollouts,
        "beta_model": args.beta_model,
        "summary": summary,
        "scored_rows": scored,
    }
    save_json(Path(args.out_json), report)

    lines = []
    lines.append("# Phase-B β-weighted Reward Report v0")
    lines.append("")
    lines.append(f"- Rollouts: `{args.rollouts}`")
    lines.append(f"- Beta model: `{args.beta_model}`")
    lines.append("")
    lines.append("| world::action | n | reach | beta_v | beta_s | beta_e | R_v | R_s | R_e | score_beta | proxy_reward | final | drift |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in sorted(summary, key=lambda x: (x["world_action"].split("::")[0], -x["mean_score_beta"])):
        lines.append(
            f"| {s['world_action']} | {s['n']} | {s['reach_rate']:.3f} | "
            f"{s['mean_beta_v']:.3f} | {s['mean_beta_s']:.3f} | {s['mean_beta_e']:.3f} | "
            f"{s['mean_R_v']:.3f} | {s['mean_R_s']:.3f} | {s['mean_R_e']:.3f} | "
            f"{s['mean_score_beta']:.3f} | {s['mean_proxy_reward']:.3f} | "
            f"{s['mean_final_dist']:.3f} | {s['mean_drift']:.3f} |"
        )

    md = "\n".join(lines) + "\n"
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md)
    print(md)
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
