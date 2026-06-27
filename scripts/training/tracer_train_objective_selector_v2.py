#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class ObjectiveSelectorMLPv2(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def split(rows, val_frac, seed):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac)) if len(rows) >= 5 else 0
    return rows[n_val:], rows[:n_val]


@torch.no_grad()
def eval_model(model, xs, ys):
    pred = model(xs)
    mse = torch.mean((pred - ys) ** 2).item()
    mae = torch.mean(torch.abs(pred - ys)).item()
    max_abs = torch.max(torch.abs(pred - ys)).item()
    return {
        "mse": mse,
        "mae": mae,
        "max_abs": max_abs,
        "pred_mean": pred.mean(dim=0).detach().cpu().tolist(),
        "target_mean": ys.mean(dim=0).detach().cpu().tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/objective_selector_dataset_v2/objective_selector_dataset_v2.jsonl")
    ap.add_argument("--out", default="artifacts/objective_selector_v2/objective_selector_v2.pt")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_jsonl(Path(args.data))
    if not rows:
        raise RuntimeError(f"no rows found: {args.data}")

    train_rows, val_rows = split(rows, args.val_frac, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(rows[0]["x"])
    output_dim = len(rows[0]["y"])

    x_all = torch.tensor([r["x"] for r in rows], dtype=torch.float32, device=device)
    mean = x_all.mean(dim=0)
    std = x_all.std(dim=0).clamp_min(1e-6)

    def tensorize(batch):
        x = torch.tensor([r["x"] for r in batch], dtype=torch.float32, device=device)
        y = torch.tensor([r["y"] for r in batch], dtype=torch.float32, device=device)
        x = (x - mean) / std
        return x, y

    model = ObjectiveSelectorMLPv2(input_dim, args.hidden, output_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        random.shuffle(train_rows)
        model.train()
        total = 0.0
        n = 0

        for i in range(0, len(train_rows), args.batch_size):
            batch = train_rows[i:i + args.batch_size]
            x, y = tensorize(batch)
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item()) * len(batch)
            n += len(batch)

        train_loss = total / max(1, n)

        model.eval()
        with torch.no_grad():
            if val_rows:
                xv, yv = tensorize(val_rows)
                val_loss = torch.mean((model(xv) - yv) ** 2).item()
            else:
                val_loss = train_loss

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 50 == 0 or ep == args.epochs:
            print(f"[epoch {ep:04d}] train_mse={train_loss:.8f} val_mse={val_loss:.8f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    xt, yt = tensorize(train_rows)
    train_metrics = eval_model(model, xt, yt)

    if val_rows:
        xv, yv = tensorize(val_rows)
        val_metrics = eval_model(model, xv, yv)
    else:
        val_metrics = None

    xa, ya = tensorize(rows)
    all_metrics = eval_model(model, xa, ya)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_type": "ObjectiveSelectorMLPv2",
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden": args.hidden,
        "feature_names": rows[0].get("feature_names", []),
        "target_names": rows[0].get("target_names", []),
        "x_mean": mean.detach().cpu().tolist(),
        "x_std": std.detach().cpu().tolist(),
        "metrics": {
            "n_rows": len(rows),
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "best_val_mse": best_val,
            "train": train_metrics,
            "val": val_metrics,
            "all": all_metrics,
        },
    }

    torch.save(ckpt, out)

    metrics_path = out.parent / "metrics.json"
    metrics_path.write_text(json.dumps(ckpt["metrics"], indent=2))

    print("[TRACER] wrote objective selector v2")
    print(json.dumps({
        "out": str(out),
        "metrics": ckpt["metrics"],
        "device": str(device),
    }, indent=2))


if __name__ == "__main__":
    main()
