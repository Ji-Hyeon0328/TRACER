#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


BOOL_LABELS = [
    "approach_success",
    "future_low_progress",
    "future_high_drift",
    "future_bad_locomotion",
    "future_good_locomotion",
]

REG_LABELS = [
    "future_risk_proxy",
]


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


class RAMEpisodeNet(nn.Module):
    def __init__(self, input_dim, hidden=96, bool_dim=5, reg_dim=1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 48),
            nn.ReLU(),
        )
        self.bool_head = nn.Linear(48, bool_dim)
        self.reg_head = nn.Linear(48, reg_dim)

    def forward(self, x):
        h = self.trunk(x)
        return self.bool_head(h), self.reg_head(h)


def collect_feature_names(rows):
    names = set()
    for r in rows:
        names.update((r.get("input_features") or {}).keys())
    return sorted(names)


def vec(feat, names):
    return torch.tensor([float(feat.get(k, 0.0)) for k in names], dtype=torch.float32)


def make_examples(rows, names):
    out = []
    for r in rows:
        lab = r.get("labels") or {}
        out.append({
            "x": vec(r["input_features"], names),
            "yb": torch.tensor([1.0 if lab.get(k) else 0.0 for k in BOOL_LABELS], dtype=torch.float32),
            "yr": torch.tensor([float(lab.get(k, 0.0)) for k in REG_LABELS], dtype=torch.float32),
            "world": r.get("world_name"),
            "action": r.get("action_name"),
        })
    return out


def eval_metrics(model, examples):
    if not examples:
        return None

    correct = 0
    total = 0
    mse_sum = 0.0

    with torch.no_grad():
        for e in examples:
            logits, reg = model(e["x"].unsqueeze(0))
            prob = torch.sigmoid(logits[0])
            pred = (prob >= 0.5).float()
            correct += int((pred == e["yb"]).sum().item())
            total += len(BOOL_LABELS)
            mse_sum += float(F.mse_loss(reg[0], e["yr"]).item())

    return {
        "bool_acc": correct / max(total, 1),
        "risk_mse": mse_sum / max(len(examples), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/phase_b_training_pipeline_v1/phase_b_ram_teacher_student_episode_labels_v1.jsonl")
    ap.add_argument("--out-dir", default="artifacts/phase_b_ram_episode_v1")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.labels)
    if len(rows) < 6:
        raise RuntimeError(f"Not enough RAM label rows: {len(rows)}")

    feature_names = collect_feature_names(rows)
    examples = make_examples(rows, feature_names)
    random.shuffle(examples)

    n_val = max(1, int(len(examples) * args.val_frac))
    val = examples[:n_val]
    train = examples[n_val:]

    model = RAMEpisodeNet(input_dim=len(feature_names))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best = {
        "epoch": -1,
        "score": -1e9,
        "state_dict": None,
    }

    for ep in range(1, args.epochs + 1):
        random.shuffle(train)
        losses = []

        model.train()
        for e in train:
            logits, reg = model(e["x"].unsqueeze(0))
            loss_bool = F.binary_cross_entropy_with_logits(logits[0], e["yb"])
            loss_reg = F.mse_loss(torch.sigmoid(reg[0]), e["yr"])
            loss = loss_bool + 1.0 * loss_reg

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        tr = eval_metrics(model, train)
        va = eval_metrics(model, val)
        mean_loss = sum(losses) / max(len(losses), 1)
        score = va["bool_acc"] - va["risk_mse"]

        row = {
            "epoch": ep,
            "loss": mean_loss,
            "train_bool_acc": tr["bool_acc"],
            "train_risk_mse": tr["risk_mse"],
            "val_bool_acc": va["bool_acc"],
            "val_risk_mse": va["risk_mse"],
            "selection_score": score,
        }
        history.append(row)

        if score > best["score"]:
            best = {
                "epoch": ep,
                "score": score,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

        if ep == 1 or ep % 50 == 0 or ep == args.epochs:
            print(json.dumps(row, sort_keys=True))

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "schema": "phase_b_ram_episode_v1",
        "model_state_dict": model.state_dict(),
        "input_dim": len(feature_names),
        "feature_names": feature_names,
        "hidden_dim": 96,
        "bool_label_names": BOOL_LABELS,
        "reg_label_names": REG_LABELS,
        "train_rows": len(train),
        "val_rows": len(val),
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "note": "Episode-level RAM teacher-student seed model. Later replace/extend with window-level proprioceptive RAM.",
    }
    torch.save(ckpt, out_dir / "model.pt")

    final_train = eval_metrics(model, train)
    final_val = eval_metrics(model, val)

    report = {
        "schema": "phase_b_ram_episode_train_report_v1",
        "labels": args.labels,
        "out_dir": str(out_dir),
        "num_rows": len(rows),
        "input_dim": len(feature_names),
        "feature_names": feature_names,
        "bool_label_names": BOOL_LABELS,
        "reg_label_names": REG_LABELS,
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "final_train": final_train,
        "final_val": final_val,
        "history_tail": history[-20:],
    }
    save_json(out_dir / "train_report.json", report)

    lines = []
    lines.append("# Phase-B RAM Episode Teacher-Student Training v1")
    lines.append("")
    lines.append(f"- Rows: `{len(rows)}`")
    lines.append(f"- Input dim: `{len(feature_names)}`")
    lines.append(f"- Best epoch: `{best['epoch']}`")
    lines.append(f"- Train bool acc: `{final_train['bool_acc']:.3f}`")
    lines.append(f"- Train risk MSE: `{final_train['risk_mse']:.4f}`")
    lines.append(f"- Val bool acc: `{final_val['bool_acc']:.3f}`")
    lines.append(f"- Val risk MSE: `{final_val['risk_mse']:.4f}`")
    lines.append("")
    lines.append("This is an episode-level RAM seed model. It should later be upgraded to window-level teacher-student RAM using time-series proprioceptive features.")
    (out_dir / "train_report.md").write_text("\n".join(lines) + "\n")

    print()
    print("[wrote]", out_dir / "model.pt")
    print("[wrote]", out_dir / "train_report.json")
    print("[wrote]", out_dir / "train_report.md")


if __name__ == "__main__":
    main()
