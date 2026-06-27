#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn


def f(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def terrain_onehot(name: str) -> List[float]:
    return [
        1.0 if name == "flat_normal" else 0.0,
        1.0 if name == "rough_mid" else 0.0,
        1.0 if name == "slope_5deg" else 0.0,
    ]


FEATURE_NAMES = [
    "terrain_flat_normal",
    "terrain_rough_mid",
    "terrain_slope_5deg",
    "duration_norm",
    "vx_over_target",
    "mpc_enable_mean",
    "body_height_delta",
    "clearance_delta",
    "gate_level_norm",
    "gate_action_norm",
    "gate_override_mean",
    "ram_run_fallen_weak",
    "ram_recovery_needed_mean",
    "success_proxy",
]


def build_x(row: dict) -> List[float]:
    terrain = str(row.get("terrain", "unknown"))
    target_vx = max(1e-6, f(row.get("target_vx", 0.05)))

    x = []
    x.extend(terrain_onehot(terrain))
    x.extend([
        min(1.0, f(row.get("duration_sec", 0.0)) / 30.0),
        max(0.0, min(2.0, f(row.get("mpc_vx_mean", 0.0)) / target_vx)),
        max(0.0, min(1.0, f(row.get("mpc_enable_mean", 0.0)))),
        f(row.get("mpc_body_height_mean", 0.295)) - 0.295,
        f(row.get("mpc_clearance_mean", 0.030)) - 0.030,
        max(0.0, min(1.0, f(row.get("gate_level_mean", 0.0)) / 2.0)),
        max(0.0, min(1.0, f(row.get("gate_action_mean", 0.0)) / 2.0)),
        max(0.0, min(1.0, f(row.get("gate_override_mean", 0.0)))),
        # RAM fallen is currently poorly calibrated, so this feature is included
        # but clipped and treated as weak signal by the model.
        max(0.0, min(1.0, f(row.get("ram_run_fallen_mean", 0.0)))),
        max(0.0, min(1.0, f(row.get("ram_recovery_needed_mean", 0.0)))),
        1.0 if str(row.get("success_proxy", "False")).lower() == "true" else 0.0,
    ])
    return x


class PreferenceRewardMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open() as fobj:
        for line in fobj:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def train_val_split(items: List[dict], val_frac: float, seed: int):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_frac)) if len(items) >= 5 else 0
    return items[n_val:], items[:n_val]


@torch.no_grad()
def pair_accuracy(model, features, pairs, mean, std, device):
    if not pairs:
        return None

    correct = 0
    margins = []
    for p in pairs:
        a = p["preferred_episode_id"]
        b = p["rejected_episode_id"]
        if a not in features or b not in features:
            continue

        xa = (torch.tensor(features[a], dtype=torch.float32, device=device) - mean) / std
        xb = (torch.tensor(features[b], dtype=torch.float32, device=device) - mean) / std
        margin = float(model(xa.unsqueeze(0))[0] - model(xb.unsqueeze(0))[0])
        margins.append(margin)
        if margin > 0:
            correct += 1

    if not margins:
        return None

    return {
        "acc": correct / len(margins),
        "n": len(margins),
        "mean_margin": sum(margins) / len(margins),
        "min_margin": min(margins),
        "max_margin": max(margins),
    }


@torch.no_grad()
def export_scores(model, rows, features, mean, std, device, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    scored = []
    for r in rows:
        eid = r["episode_id"]
        x = (torch.tensor(features[eid], dtype=torch.float32, device=device) - mean) / std
        score = float(model(x.unsqueeze(0))[0])
        rr = dict(r)
        rr["pred_preference_score_v2"] = score
        scored.append(rr)

    scored.sort(key=lambda r: (r.get("terrain", ""), -f(r.get("pred_preference_score_v2", 0.0))))

    if scored:
        with out_csv.open("w", newline="") as fobj:
            w = csv.DictWriter(fobj, fieldnames=list(scored[0].keys()))
            w.writeheader()
            w.writerows(scored)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="data/preference_datasets_v2/rollout_metrics_v2.jsonl")
    ap.add_argument("--pairs", default="data/preference_datasets_v2/rollout_preference_pairs_v2.jsonl")
    ap.add_argument("--out", default="artifacts/objective_preference_v2/objective_preference_v2.pt")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    metrics_path = Path(args.metrics)
    pairs_path = Path(args.pairs)
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(metrics_path)
    pairs = load_jsonl(pairs_path)

    if not rows:
        raise RuntimeError(f"no metric rows found: {metrics_path}")
    if not pairs:
        raise RuntimeError(f"no preference pairs found: {pairs_path}")

    features: Dict[str, List[float]] = {}
    row_by_id = {}
    for r in rows:
        eid = r["episode_id"]
        features[eid] = build_x(r)
        row_by_id[eid] = r

    usable_pairs = [
        p for p in pairs
        if p["preferred_episode_id"] in features and p["rejected_episode_id"] in features
    ]

    train_pairs, val_pairs = train_val_split(usable_pairs, args.val_frac, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_all = torch.tensor(list(features.values()), dtype=torch.float32, device=device)
    mean = x_all.mean(dim=0)
    std = x_all.std(dim=0).clamp_min(1e-6)

    model = PreferenceRewardMLP(input_dim=len(FEATURE_NAMES), hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    def make_batch(batch_pairs: List[dict]):
        xa, xb = [], []
        for p in batch_pairs:
            xa.append(features[p["preferred_episode_id"]])
            xb.append(features[p["rejected_episode_id"]])
        xa = torch.tensor(xa, dtype=torch.float32, device=device)
        xb = torch.tensor(xb, dtype=torch.float32, device=device)
        xa = (xa - mean) / std
        xb = (xb - mean) / std
        y = torch.ones((xa.shape[0],), dtype=torch.float32, device=device)
        return xa, xb, y

    best_val = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        random.shuffle(train_pairs)
        model.train()
        total_loss = 0.0
        n_seen = 0

        for i in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[i:i + args.batch_size]
            xa, xb, y = make_batch(batch)
            logits = model(xa) - model(xb)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(batch)
            n_seen += len(batch)

        train_loss = total_loss / max(1, n_seen)

        model.eval()
        if val_pairs:
            with torch.no_grad():
                xa, xb, y = make_batch(val_pairs)
                val_loss = float(loss_fn(model(xa) - model(xb), y).item())
        else:
            val_loss = train_loss

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 50 == 0 or ep == args.epochs:
            train_acc = pair_accuracy(model, features, train_pairs, mean, std, device)
            val_acc = pair_accuracy(model, features, val_pairs, mean, std, device) if val_pairs else None
            print(
                f"[epoch {ep:04d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"train_acc={train_acc['acc'] if train_acc else None} "
                f"val_acc={val_acc['acc'] if val_acc else None}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = pair_accuracy(model, features, train_pairs, mean, std, device)
    val_metrics = pair_accuracy(model, features, val_pairs, mean, std, device) if val_pairs else None
    all_metrics = pair_accuracy(model, features, usable_pairs, mean, std, device)

    ckpt = {
        "model_type": "PreferenceRewardMLP",
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": len(FEATURE_NAMES),
        "hidden": args.hidden,
        "feature_names": FEATURE_NAMES,
        "x_mean": mean.detach().cpu().tolist(),
        "x_std": std.detach().cpu().tolist(),
        "metrics": {
            "n_rows": len(rows),
            "n_pairs": len(usable_pairs),
            "n_train_pairs": len(train_pairs),
            "n_val_pairs": len(val_pairs),
            "best_val_loss": best_val,
            "train": train_metrics,
            "val": val_metrics,
            "all": all_metrics,
        },
    }

    torch.save(ckpt, out_path)

    metrics_json = out_dir / "metrics.json"
    metrics_json.write_text(json.dumps(ckpt["metrics"], indent=2))

    scored_csv = out_dir / "scored_episodes_v2.csv"
    export_scores(model, rows, features, mean, std, device, scored_csv)

    print("[TRACER] wrote objective preference model v2")
    print(json.dumps({
        "out": str(out_path),
        "metrics_json": str(metrics_json),
        "scored_csv": str(scored_csv),
        "device": str(device),
        "metrics": ckpt["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
