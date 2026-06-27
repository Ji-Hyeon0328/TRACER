#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ObjectiveSelectorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        # Softmax keeps beta normalized and positive.
        return torch.softmax(self.net(x), dim=-1)


def load_data(path):
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            xs.append([float(v) for v in o["x"]])
            ys.append([float(v) for v in o["beta"]])

    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)

    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0).clamp_min(1e-6)
    x = (x - x_mean) / x_std

    return x, y, x_mean, x_std


def split(n, val_frac, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    nv = max(1, int(round(n * val_frac)))
    return idx[nv:], idx[:nv]


def eval_model(model, loader, device):
    model.eval()
    total = 0
    mse_sum = 0.0
    mae_sum = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse = ((pred - y) ** 2).mean(dim=1)
            mae = (pred - y).abs().mean(dim=1)
            mse_sum += float(mse.sum().item())
            mae_sum += float(mae.sum().item())
            total += x.shape[0]

    return {
        "mse": mse_sum / max(1, total),
        "mae": mae_sum / max(1, total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/objective_selector_dataset_v1/objective_selector_dataset_v1.jsonl")
    ap.add_argument("--out", default="artifacts/objective_selector_v1/objective_selector_v1.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.25)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    x, y, x_mean, x_std = load_data(args.data)
    tr_idx, va_idx = split(len(y), args.val_frac, args.seed)

    tr_loader = DataLoader(TensorDataset(x[tr_idx], y[tr_idx]), batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(x[va_idx], y[va_idx]), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ObjectiveSelectorMLP(x.shape[1], args.hidden).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best = None
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if ep == 1 or ep % 25 == 0 or ep == args.epochs:
            trm = eval_model(model, tr_loader, device)
            vam = eval_model(model, va_loader, device)
            if best is None or vam["mse"] < best:
                best = vam["mse"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            print(
                f"epoch={ep:04d} "
                f"train_mse={trm['mse']:.8f} train_mae={trm['mae']:.6f} "
                f"val_mse={vam['mse']:.8f} val_mae={vam['mae']:.6f}"
            )

    if best_state:
        model.load_state_dict(best_state)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model_state_dict": model.cpu().state_dict(),
        "input_dim": int(x.shape[1]),
        "hidden_dim": args.hidden,
        "output_dim": 3,
        "beta_names": ["motion", "stability", "energy"],
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "data": args.data,
        "train_metrics": eval_model(model, tr_loader, torch.device("cpu")),
        "val_metrics": eval_model(model, va_loader, torch.device("cpu")),
    }

    torch.save(bundle, out)

    metrics = {
        "train": bundle["train_metrics"],
        "val": bundle["val_metrics"],
        "n_samples": int(len(y)),
        "input_dim": int(x.shape[1]),
        "beta_names": ["motion", "stability", "energy"],
    }

    out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[TRACER] saved: {out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
