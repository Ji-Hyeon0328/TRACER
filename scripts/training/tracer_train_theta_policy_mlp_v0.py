#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ThetaPolicyMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def load_dataset(path: Path):
    xs, ys = [], []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            xs.append(r["x"])
            ys.append(r["theta"])
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/meta_gait_theta_dataset/theta_policy_warmstart_v0.jsonl")
    ap.add_argument("--out", default="artifacts/theta_policy_mlp_v0/theta_policy_mlp_v0.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x, y = load_dataset(data_path)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    model = ThetaPolicyMLP(input_dim=x.shape[1], output_dim=y.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for ep in range(1, args.epochs + 1):
        total = 0.0
        n = 0
        for xb, yb in dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]

        if ep == 1 or ep % 25 == 0 or ep == args.epochs:
            print(f"[TRACER] epoch={ep:04d} loss={total / max(1, n):.6f}")

    artifact = {
        "model_state_dict": model.state_dict(),
        "input_dim": int(x.shape[1]),
        "output_dim": int(y.shape[1]),
        "dataset": str(data_path),
        "note": "BC-style warm-start theta policy MLP v0. Outputs normalized theta in [-1,1].",
    }
    torch.save(artifact, out_path)
    print(f"[TRACER] saved {out_path}")

    with torch.no_grad():
        pred = model(x[:5])
        print("[TRACER] sample pred:")
        print(pred)


if __name__ == "__main__":
    main()
