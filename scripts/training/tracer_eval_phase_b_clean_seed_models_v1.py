#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


class UtilityNet(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RAMEpisodeNet(nn.Module):
    def __init__(self, input_dim, hidden=64, bool_dim=5, reg_dim=1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
        )
        self.bool_head = nn.Linear(32, bool_dim)
        self.risk_head = nn.Linear(32, reg_dim)

    def forward(self, x):
        h = self.trunk(x)
        return self.bool_head(h), torch.sigmoid(self.risk_head(h))


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


def vec(feat, names):
    return torch.tensor([float((feat or {}).get(k, 0.0)) for k in names], dtype=torch.float32)


def fmt(x):
    return f"{float(x):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="data/phase_b_training_pipeline_v1/phase_b_rollout_index_v1.jsonl")
    ap.add_argument("--objective-model", default="artifacts/phase_b_objective_selector_pref_v2_clean/model.pt")
    ap.add_argument("--ram-model", default="artifacts/phase_b_ram_episode_v2_clean/model.pt")
    ap.add_argument("--out-json", default="reports/phase_b_clean_seed_model_eval_v1.json")
    ap.add_argument("--out-md", default="reports/phase_b_clean_seed_model_eval_v1.md")
    args = ap.parse_args()

    rows = read_jsonl(args.rollouts)

    obj_ckpt = torch.load(args.objective_model, map_location="cpu", weights_only=False)
    ram_ckpt = torch.load(args.ram_model, map_location="cpu", weights_only=False)

    obj = UtilityNet(obj_ckpt["input_dim"], hidden=obj_ckpt.get("hidden_dim", 64))
    obj.load_state_dict(obj_ckpt["model_state_dict"])
    obj.eval()

    ram = RAMEpisodeNet(
        ram_ckpt["input_dim"],
        hidden=ram_ckpt.get("hidden_dim", 64),
        bool_dim=len(ram_ckpt["bool_label_names"]),
        reg_dim=len(ram_ckpt["reg_label_names"]),
    )
    ram.load_state_dict(ram_ckpt["model_state_dict"])
    ram.eval()

    obj_names = obj_ckpt["feature_names"]
    ram_names = ram_ckpt["feature_names"]

    grouped = {}

    with torch.no_grad():
        for r in rows:
            world = r["world_name"]
            action = r["action_name"]
            key = f"{world}::{action}"

            ox = vec(r["features"], obj_names).unsqueeze(0)
            rx = vec(r["features"], ram_names).unsqueeze(0)

            utility = float(obj(ox)[0].item())
            bool_logits, risk = ram(rx)
            risk_v = float(risk[0, 0].item())

            g = grouped.setdefault(key, {
                "world": world,
                "action": action,
                "n": 0,
                "utility_sum": 0.0,
                "risk_sum": 0.0,
                "seed_score_sum": 0.0,
                "seed_risk_sum": 0.0,
                "reach_count": 0,
            })
            g["n"] += 1
            g["utility_sum"] += utility
            g["risk_sum"] += risk_v
            g["seed_score_sum"] += float(r["preference_score_seed"])
            g["seed_risk_sum"] += float(r["ram_labels"]["future_risk_proxy"])
            g["reach_count"] += 1 if r["reward_components"].get("reached_stop_distance") else 0

    rows_out = []
    for key, g in grouped.items():
        n = max(g["n"], 1)
        rows_out.append({
            "world": g["world"],
            "action": g["action"],
            "n": g["n"],
            "pred_utility": g["utility_sum"] / n,
            "pred_risk": g["risk_sum"] / n,
            "seed_score": g["seed_score_sum"] / n,
            "seed_risk": g["seed_risk_sum"] / n,
            "reach_rate": g["reach_count"] / n,
        })

    rows_out = sorted(rows_out, key=lambda x: (x["world"], -x["pred_utility"]))

    report = {
        "schema": "phase_b_clean_seed_model_eval_v1",
        "objective_model": args.objective_model,
        "ram_model": args.ram_model,
        "rows": rows_out,
    }
    save_json(Path(args.out_json), report)

    lines = []
    lines.append("# Phase-B Clean Seed Model Evaluation v1")
    lines.append("")
    lines.append("| world | action | n | pred_utility | seed_score | pred_risk | seed_risk | reach_rate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows_out:
        lines.append(
            f"| {r['world']} | {r['action']} | {r['n']} | {fmt(r['pred_utility'])} | "
            f"{fmt(r['seed_score'])} | {fmt(r['pred_risk'])} | {fmt(r['seed_risk'])} | {fmt(r['reach_rate'])} |"
        )

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
