#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class GMSMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_data(path):
    xs, ys, labels = [], [], None
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            xs.append([float(v) for v in o["x"]])
            ys.append(int(o["label_id"]))
            labels = o["label_names"]

    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)

    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0).clamp_min(1e-6)
    x = (x - x_mean) / x_std

    return x, y, labels, x_mean, x_std


def split(n, val_frac, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    nv = max(1, int(round(n * val_frac)))
    return idx[nv:], idx[:nv]


def eval_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            loss_sum += float(loss.item()) * y.numel()

    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/gms_dataset_v1/gms_dataset_v1.jsonl")
    ap.add_argument("--out", default="artifacts/gms_classifier_v1/gms_classifier_v1.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    x, y, labels, x_mean, x_std = load_data(args.data)
    tr_idx, va_idx = split(len(y), args.val_frac, args.seed)

    tr = TensorDataset(x[tr_idx], y[tr_idx])
    va = TensorDataset(x[va_idx], y[va_idx])

    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GMSMLP(x.shape[1], len(labels), args.hidden).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    best = None
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if ep == 1 or ep % 25 == 0 or ep == args.epochs:
            trm = eval_model(model, tr_loader, device)
            vam = eval_model(model, va_loader, device)
            if best is None or vam["loss"] < best:
                best = vam["loss"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(
                f"epoch={ep:04d} "
                f"train_loss={trm['loss']:.6f} train_acc={trm['acc']:.3f} "
                f"val_loss={vam['loss']:.6f} val_acc={vam['acc']:.3f}"
            )

    if best_state:
        model.load_state_dict(best_state)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model_state_dict": model.cpu().state_dict(),
        "input_dim": int(x.shape[1]),
        "hidden_dim": args.hidden,
        "label_names": labels,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "data": args.data,
        "train_metrics": eval_model(model, tr_loader, torch.device("cpu")),
        "val_metrics": eval_model(model, va_loader, torch.device("cpu")),
    }

    torch.save(bundle, out)
    out.with_suffix(".metrics.json").write_text(json.dumps({
        "train": bundle["train_metrics"],
        "val": bundle["val_metrics"],
        "label_names": labels,
        "n_samples": int(len(y)),
        "input_dim": int(x.shape[1]),
    }, indent=2))

    print(f"[TRACER] saved: {out}")
    print(json.dumps({"val": bundle["val_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
