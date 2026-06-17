#!/usr/bin/env python3

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ProprioEncoderMLP(nn.Module):
    def __init__(self, input_dim=45, output_dim=32, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def decode(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def load_npz_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(os.path.join(data_dir, "tracer_dataset_*.npz")))

    if not files:
        raise FileNotFoundError(f"No tracer_dataset_*.npz found in {data_dir}")

    xs = []
    ys = []

    for path in files:
        d = np.load(path)

        proprio = d["proprio"]
        context = d["context"]
        latent = d["latent"]
        valid = d["valid_mask"] > 0.5

        x = proprio[valid]
        y = np.concatenate([context[valid], latent[valid]], axis=1)

        finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x = x[finite]
        y = y[finite]

        xs.append(x)
        ys.append(y)

        print(f"[load] {os.path.basename(path)} x={x.shape} y={y.shape}")

    X = np.concatenate(xs, axis=0).astype(np.float32)
    Y = np.concatenate(ys, axis=0).astype(np.float32)

    print(f"[dataset] X={X.shape}, Y={Y.shape}")
    return X, Y


def make_normalizer(x: np.ndarray, eps: float = 1e-6) -> Normalizer:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.maximum(std, eps)
    return Normalizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)

    X, Y = load_npz_dataset(args.data_dir)

    x_norm = make_normalizer(X)
    y_norm = make_normalizer(Y)

    Xn = x_norm.encode(X).astype(np.float32)
    Yn = y_norm.encode(Y).astype(np.float32)

    n = Xn.shape[0]
    idx = np.random.default_rng(args.seed).permutation(n)

    n_train = int(n * args.train_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    X_train = torch.from_numpy(Xn[train_idx])
    Y_train = torch.from_numpy(Yn[train_idx])

    X_val = torch.from_numpy(Xn[val_idx])
    Y_val = torch.from_numpy(Yn[val_idx])

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[device] {device}")

    model = ProprioEncoderMLP(
        input_dim=X.shape[1],
        output_dim=Y.shape[1],
        hidden_dim=args.hidden_dim,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "tracer_proprio_encoder_v0_best.pt")
    final_path = os.path.join(args.out_dir, "tracer_proprio_encoder_v0_final.pt")
    ts_path = os.path.join(args.out_dir, "tracer_proprio_encoder_v0.ts")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss

        if val_loss < best_val:
            best_val = val_loss

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": X.shape[1],
                    "output_dim": Y.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "x_mean": x_norm.mean,
                    "x_std": x_norm.std,
                    "y_mean": y_norm.mean,
                    "y_std": y_norm.std,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "data_dir": args.data_dir,
                },
                best_path,
            )

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"[epoch {epoch:04d}] "
                f"train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}"
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": X.shape[1],
            "output_dim": Y.shape[1],
            "hidden_dim": args.hidden_dim,
            "x_mean": x_norm.mean,
            "x_std": x_norm.std,
            "y_mean": y_norm.mean,
            "y_std": y_norm.std,
            "epoch": args.epochs,
            "data_dir": args.data_dir,
        },
        final_path,
    )

    model_cpu = model.to("cpu").eval()
    example = torch.zeros(1, X.shape[1], dtype=torch.float32)
    traced = torch.jit.trace(model_cpu, example)
    traced.save(ts_path)

    print(f"[saved] best checkpoint:  {best_path}")
    print(f"[saved] final checkpoint: {final_path}")
    print(f"[saved] torchscript:      {ts_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", default=os.path.expanduser("~/Tracer/TRACER/data"))
    parser.add_argument("--out-dir", default=os.path.expanduser("~/Tracer/TRACER/checkpoints"))

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--print-every", type=int, default=10)

    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
