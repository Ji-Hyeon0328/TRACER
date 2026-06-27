#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


class RamShadowMLPv2(nn.Module):
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


def split(rows, val_frac, seed):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac)) if len(rows) >= 5 else 0
    return rows[n_val:], rows[:n_val]


@torch.no_grad()
def eval_model(model, x, y, binary_idx, regression_idx):
    logits = model(x)
    metrics = {}

    if binary_idx:
        pred = torch.sigmoid(logits[:, binary_idx])
        target = y[:, binary_idx]
        acc = ((pred >= 0.5) == (target >= 0.5)).float().mean(dim=0)
        metrics["binary_acc_mean"] = float(acc.mean().item())
        metrics["binary_acc_per_label"] = acc.detach().cpu().tolist()

    if regression_idx:
        # Regression heads are trained through sigmoid because labels are in [0, 1].
        # Report MAE in the same bounded probability space, not raw logit space.
        pred_r = torch.sigmoid(logits[:, regression_idx])
        target_r = y[:, regression_idx]
        mae = torch.abs(pred_r - target_r).mean(dim=0)
        metrics["reg_mae_mean"] = float(mae.mean().item())
        metrics["reg_mae_per_label"] = mae.detach().cpu().tolist()

    mse = torch.mean((torch.sigmoid(logits) - y) ** 2).item()
    metrics["sigmoid_mse_all"] = mse
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ram_shadow_dataset_v2/ram_shadow_windows_v2.jsonl")
    ap.add_argument("--out", default="artifacts/ram_shadow_v2/ram_shadow_v2.pt")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_jsonl(Path(args.data))
    if not rows:
        raise RuntimeError(f"no rows found: {args.data}")

    label_names = rows[0]["label_names"]
    regression_names = {"future_override_mean"}
    regression_idx = [i for i, n in enumerate(label_names) if n in regression_names]
    binary_idx = [i for i, n in enumerate(label_names) if n not in regression_names]

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

    model = RamShadowMLPv2(input_dim, output_dim, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

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
            out = model(x)

            loss = 0.0
            if binary_idx:
                loss = loss + bce(out[:, binary_idx], y[:, binary_idx])
            if regression_idx:
                loss = loss + mse(torch.sigmoid(out[:, regression_idx]), y[:, regression_idx])

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
                ov = model(xv)
                val_loss = 0.0
                if binary_idx:
                    val_loss = val_loss + float(bce(ov[:, binary_idx], yv[:, binary_idx]).item())
                if regression_idx:
                    val_loss = val_loss + float(mse(torch.sigmoid(ov[:, regression_idx]), yv[:, regression_idx]).item())
            else:
                val_loss = train_loss

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 50 == 0 or ep == args.epochs:
            print(f"[epoch {ep:04d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    xt, yt = tensorize(train_rows)
    train_metrics = eval_model(model, xt, yt, binary_idx, regression_idx)

    if val_rows:
        xv, yv = tensorize(val_rows)
        val_metrics = eval_model(model, xv, yv, binary_idx, regression_idx)
    else:
        val_metrics = None

    xa, ya = tensorize(rows)
    all_metrics = eval_model(model, xa, ya, binary_idx, regression_idx)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_type": "RamShadowMLPv2",
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": args.hidden,
        "feature_names": rows[0].get("feature_names", []),
        "label_names": label_names,
        "x_mean": mean.detach().cpu().tolist(),
        "x_std": std.detach().cpu().tolist(),
        "metrics": {
            "n_rows": len(rows),
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "best_val_loss": best_val,
            "binary_idx": binary_idx,
            "regression_idx": regression_idx,
            "train": train_metrics,
            "val": val_metrics,
            "all": all_metrics,
        },
        "notes": "RAM shadow v2 warm-start model trained from supervised-stack shadow logs, not true privileged physics labels.",
    }

    torch.save(ckpt, out_path)

    metrics_path = out_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(ckpt["metrics"], indent=2))

    print("[TRACER] wrote RAM shadow model v2")
    print(json.dumps({
        "out": str(out_path),
        "metrics_json": str(metrics_path),
        "device": str(device),
        "metrics": ckpt["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
