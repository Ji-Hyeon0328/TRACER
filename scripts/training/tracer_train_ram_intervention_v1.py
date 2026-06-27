#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


LABELS = [
    "future_gate_caution",
    "future_gate_unstable",
    "future_conservative_probe",
    "future_override_mean",
    "future_low_speed",
    "future_disabled",
    "episode_success",
]


class RamWindowDataset(Dataset):
    def __init__(self, path: str):
        self.items = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    x = [float(v) for v in obj["x"]]
                    y = [float(obj["y"].get(k, 0.0)) for k in LABELS]
                    self.items.append({
                        "x": x,
                        "y": y,
                        "terrain": obj.get("terrain", "unknown"),
                        "episode_id": obj.get("episode_id", "unknown"),
                    })

        if not self.items:
            raise RuntimeError(f"empty dataset: {path}")

        self.input_dim = len(self.items[0]["x"])
        for it in self.items:
            if len(it["x"]) != self.input_dim:
                raise RuntimeError("inconsistent input dimensions")

        xs = torch.tensor([it["x"] for it in self.items], dtype=torch.float32)
        self.x_mean = xs.mean(dim=0)
        self.x_std = xs.std(dim=0).clamp_min(1e-6)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        x = torch.tensor(it["x"], dtype=torch.float32)
        x = (x - self.x_mean) / self.x_std
        y = torch.tensor(it["y"], dtype=torch.float32)
        return x, y


class RamInterventionMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256):
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
        return self.net(x)


def split_indices(n: int, val_frac: float, seed: int):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    val = idx[:n_val]
    train = idx[n_val:]
    return train, val


def subset_loader(ds, indices, batch_size, shuffle):
    xs = []
    ys = []
    for i in indices:
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    tensor_ds = torch.utils.data.TensorDataset(x, y)
    return DataLoader(tensor_ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, bce_loss, mse_loss, device):
    model.eval()
    total = 0
    loss_sum = 0.0

    y_all = []
    p_all = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            # First, second, third, fifth, sixth, seventh are binary-ish.
            bidx = [0, 1, 2, 4, 5, 6]
            midx = [3]

            loss = bce_loss(logits[:, bidx], y[:, bidx])
            loss = loss + mse_loss(torch.sigmoid(logits[:, midx]), y[:, midx])

            loss_sum += float(loss.item()) * x.shape[0]
            total += x.shape[0]

            y_all.append(y.cpu())
            p_all.append(torch.sigmoid(logits).cpu())

    y_all = torch.cat(y_all, dim=0)
    p_all = torch.cat(p_all, dim=0)

    metrics = {"loss": loss_sum / max(1, total)}
    for j, name in enumerate(LABELS):
        y = y_all[:, j]
        p = p_all[:, j]
        pred = (p >= 0.5).float()
        acc = (pred == (y >= 0.5).float()).float().mean().item()
        mae = (p - y).abs().mean().item()
        metrics[f"{name}_acc"] = acc
        metrics[f"{name}_mae"] = mae
        metrics[f"{name}_target_mean"] = y.mean().item()
        metrics[f"{name}_pred_mean"] = p.mean().item()

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ram_window_dataset_v1/ram_windows_v1.jsonl")
    ap.add_argument("--out", default="artifacts/ram_intervention_v1/ram_intervention_v1.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ds = RamWindowDataset(args.data)
    train_idx, val_idx = split_indices(len(ds), args.val_frac, args.seed)

    train_loader = subset_loader(ds, train_idx, args.batch_size, True)
    val_loader = subset_loader(ds, val_idx, args.batch_size, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RamInterventionMLP(ds.input_dim, len(LABELS), args.hidden).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    best = None
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            bidx = [0, 1, 2, 4, 5, 6]
            midx = [3]

            loss = bce_loss(logits[:, bidx], y[:, bidx])
            loss = loss + mse_loss(torch.sigmoid(logits[:, midx]), y[:, midx])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if ep == 1 or ep % 25 == 0 or ep == args.epochs:
            tr = evaluate(model, train_loader, bce_loss, mse_loss, device)
            va = evaluate(model, val_loader, bce_loss, mse_loss, device)

            if best is None or va["loss"] < best:
                best = va["loss"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            print(
                f"epoch={ep:04d} "
                f"train_loss={tr['loss']:.6f} "
                f"val_loss={va['loss']:.6f} "
                f"val_caution_acc={va['future_gate_caution_acc']:.3f} "
                f"val_unstable_acc={va['future_gate_unstable_acc']:.3f} "
                f"val_conservative_acc={va['future_conservative_probe_acc']:.3f} "
                f"val_low_speed_acc={va['future_low_speed_acc']:.3f}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_loader, bce_loss, mse_loss, device)
    val_metrics = evaluate(model, val_loader, bce_loss, mse_loss, device)

    bundle = {
        "model_state_dict": model.cpu().state_dict(),
        "input_dim": ds.input_dim,
        "output_dim": len(LABELS),
        "hidden_dim": args.hidden,
        "label_names": LABELS,
        "x_mean": ds.x_mean.tolist(),
        "x_std": ds.x_std.tolist(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "data": args.data,
    }

    torch.save(bundle, out)

    metrics_path = out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({
        "train": train_metrics,
        "val": val_metrics,
        "n_samples": len(ds),
        "input_dim": ds.input_dim,
        "labels": LABELS,
    }, indent=2))

    print(f"[TRACER] saved: {out}")
    print(f"[TRACER] metrics: {metrics_path}")
    print(json.dumps({"val": val_metrics}, indent=2))


if __name__ == "__main__":
    main()
