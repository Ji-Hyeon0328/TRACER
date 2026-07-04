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


class UtilityNet(nn.Module):
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


def collect_feature_names(rows):
    names = set()
    for r in rows:
        names.update((r.get("winner_features") or {}).keys())
        names.update((r.get("loser_features") or {}).keys())
    return sorted(names)


def vec(feat, names):
    return torch.tensor([float(feat.get(k, 0.0)) for k in names], dtype=torch.float32)


def make_examples(rows, names):
    out = []
    for r in rows:
        out.append({
            "winner": vec(r["winner_features"], names),
            "loser": vec(r["loser_features"], names),
            "world": r.get("world_name"),
            "winner_action": r.get("winner_action"),
            "loser_action": r.get("loser_action"),
            "gap": float(r.get("score_gap", 0.0)),
        })
    return out


def eval_acc(model, examples):
    if not examples:
        return None
    ok = 0
    with torch.no_grad():
        for e in examples:
            sw = model(e["winner"].unsqueeze(0))[0].item()
            sl = model(e["loser"].unsqueeze(0))[0].item()
            ok += 1 if sw > sl else 0
    return ok / len(examples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/phase_b_training_pipeline_v1/phase_b_objective_preference_pairs_v1.jsonl")
    ap.add_argument("--out-dir", default="artifacts/phase_b_objective_selector_pref_v1")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.pairs)
    if len(rows) < 4:
        raise RuntimeError(f"Not enough preference pairs: {len(rows)}")

    feature_names = collect_feature_names(rows)
    examples = make_examples(rows, feature_names)
    random.shuffle(examples)

    n_val = max(1, int(len(examples) * args.val_frac))
    val = examples[:n_val]
    train = examples[n_val:]

    model = UtilityNet(input_dim=len(feature_names))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best = {
        "epoch": -1,
        "val_acc": -1.0,
        "state_dict": None,
    }

    for ep in range(1, args.epochs + 1):
        random.shuffle(train)
        losses = []

        model.train()
        for e in train:
            sw = model(e["winner"].unsqueeze(0))
            sl = model(e["loser"].unsqueeze(0))
            # Preference loss: winner utility should exceed loser utility.
            loss = F.softplus(-(sw - sl)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        train_acc = eval_acc(model, train)
        val_acc = eval_acc(model, val)
        mean_loss = sum(losses) / max(len(losses), 1)

        row = {
            "epoch": ep,
            "loss": mean_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(row)

        if val_acc is not None and val_acc > best["val_acc"]:
            best = {
                "epoch": ep,
                "val_acc": val_acc,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

        if ep == 1 or ep % 50 == 0 or ep == args.epochs:
            print(json.dumps(row, sort_keys=True))

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "schema": "phase_b_objective_selector_pref_v1",
        "model_state_dict": model.state_dict(),
        "input_dim": len(feature_names),
        "feature_names": feature_names,
        "hidden_dim": 96,
        "train_pairs": len(train),
        "val_pairs": len(val),
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "note": "Bootstrap preference/self-IRL Objective Selector utility model. Scores terrain/action/outcome feature vectors.",
    }
    torch.save(ckpt, out_dir / "model.pt")

    report = {
        "schema": "phase_b_objective_selector_pref_train_report_v1",
        "pairs": args.pairs,
        "out_dir": str(out_dir),
        "num_pairs": len(rows),
        "input_dim": len(feature_names),
        "feature_names": feature_names,
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "final_train_acc": eval_acc(model, train),
        "final_val_acc": eval_acc(model, val),
        "history_tail": history[-20:],
    }
    save_json(out_dir / "train_report.json", report)

    lines = []
    lines.append("# Phase-B Objective Selector Preference Training v1")
    lines.append("")
    lines.append(f"- Pairs: `{len(rows)}`")
    lines.append(f"- Input dim: `{len(feature_names)}`")
    lines.append(f"- Best epoch: `{best['epoch']}`")
    lines.append(f"- Best val acc: `{best['val_acc']:.3f}`")
    lines.append(f"- Final train acc: `{report['final_train_acc']:.3f}`")
    lines.append(f"- Final val acc: `{report['final_val_acc']:.3f}`")
    lines.append("")
    lines.append("This is a seed Objective Selector utility model trained from Phase-B bootstrap preference pairs.")
    (out_dir / "train_report.md").write_text("\n".join(lines) + "\n")

    print()
    print("[wrote]", out_dir / "model.pt")
    print("[wrote]", out_dir / "train_report.json")
    print("[wrote]", out_dir / "train_report.md")


if __name__ == "__main__":
    main()
