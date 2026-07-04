#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ALLOWED_PREFIXES = (
    "world_onehot:",
    "terrain_",
    "theta:",
)

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


def causal_filter(feat):
    return {
        k: float(v)
        for k, v in (feat or {}).items()
        if k.startswith(ALLOWED_PREFIXES)
    }


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
        bool_logits = self.bool_head(h)
        risk_logit = self.risk_head(h)
        risk_prob = torch.sigmoid(risk_logit)
        return bool_logits, risk_prob


def collect_feature_names(rows):
    names = set()
    excluded = set()
    for r in rows:
        feat = r.get("input_features") or {}
        for k in feat.keys():
            if k.startswith(ALLOWED_PREFIXES):
                names.add(k)
            else:
                excluded.add(k)
    return sorted(names), sorted(excluded)


def vec(feat, names):
    f = causal_filter(feat)
    return torch.tensor([float(f.get(k, 0.0)) for k in names], dtype=torch.float32)


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
            logits, risk = model(e["x"].unsqueeze(0))
            prob = torch.sigmoid(logits[0])
            pred = (prob >= 0.5).float()
            correct += int((pred == e["yb"]).sum().item())
            total += len(BOOL_LABELS)
            mse_sum += float(F.mse_loss(risk[0], e["yr"]).item())

    return {
        "bool_acc": correct / max(total, 1),
        "risk_mse": mse_sum / max(len(examples), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/phase_b_training_pipeline_v1/phase_b_ram_teacher_student_episode_labels_v1.jsonl")
    ap.add_argument("--out-dir", default="artifacts/phase_b_ram_episode_v2_clean")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.labels)
    if len(rows) < 6:
        raise RuntimeError(f"Not enough RAM label rows: {len(rows)}")

    feature_names, excluded_features = collect_feature_names(rows)
    if not feature_names:
        raise RuntimeError("No causal features found.")

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
            logits, risk = model(e["x"].unsqueeze(0))
            loss_bool = F.binary_cross_entropy_with_logits(logits[0], e["yb"])
            loss_risk = F.mse_loss(risk[0], e["yr"])
            loss = loss_bool + loss_risk

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

    final_train = eval_metrics(model, train)
    final_val = eval_metrics(model, val)

    ckpt = {
        "schema": "phase_b_ram_episode_v2_clean",
        "model_state_dict": model.state_dict(),
        "input_dim": len(feature_names),
        "feature_names": feature_names,
        "excluded_features": excluded_features,
        "allowed_prefixes": list(ALLOWED_PREFIXES),
        "hidden_dim": 64,
        "bool_label_names": BOOL_LABELS,
        "reg_label_names": REG_LABELS,
        "train_rows": len(train),
        "val_rows": len(val),
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "final_train": final_train,
        "final_val": final_val,
        "note": "Clean episode-level RAM seed model. Inputs exclude post-outcome metric/reward leakage. Risk head outputs sigmoid-bounded [0,1].",
    }
    torch.save(ckpt, out_dir / "model.pt")

    report = {
        "schema": "phase_b_ram_episode_train_report_v2_clean",
        "labels": args.labels,
        "out_dir": str(out_dir),
        "num_rows": len(rows),
        "input_dim": len(feature_names),
        "feature_names": feature_names,
        "excluded_features": excluded_features,
        "bool_label_names": BOOL_LABELS,
        "reg_label_names": REG_LABELS,
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "final_train": final_train,
        "final_val": final_val,
        "history_tail": history[-20:],
        "warning": "Episode-level RAM is a seed model. Later upgrade to window-level proprioceptive teacher-student RAM.",
    }
    save_json(out_dir / "train_report.json", report)

    lines = []
    lines.append("# Phase-B RAM Episode Teacher-Student Training v2 Clean")
    lines.append("")
    lines.append(f"- Rows: `{len(rows)}`")
    lines.append(f"- Input dim: `{len(feature_names)}`")
    lines.append(f"- Excluded leakage features: `{len(excluded_features)}`")
    lines.append(f"- Best epoch: `{best['epoch']}`")
    lines.append(f"- Train bool acc: `{final_train['bool_acc']:.3f}`")
    lines.append(f"- Train risk MSE: `{final_train['risk_mse']:.4f}`")
    lines.append(f"- Val bool acc: `{final_val['bool_acc']:.3f}`")
    lines.append(f"- Val risk MSE: `{final_val['risk_mse']:.4f}`")
    lines.append("")
    lines.append("Inputs are restricted to world/terrain/theta features. Risk prediction is sigmoid-bounded to `[0,1]` during both train and eval.")
    lines.append("")
    lines.append("This is an episode-level seed RAM. It should later be upgraded to window-level teacher-student RAM using time-series proprioception.")
    (out_dir / "train_report.md").write_text("\n".join(lines) + "\n")

    print()
    print("[wrote]", out_dir / "model.pt")
    print("[wrote]", out_dir / "train_report.json")
    print("[wrote]", out_dir / "train_report.md")


if __name__ == "__main__":
    main()
