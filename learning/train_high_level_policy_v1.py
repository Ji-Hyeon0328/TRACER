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


class HighLevelPolicyV1MLP(nn.Module):
    def __init__(self, input_dim=39, output_dim=5, hidden_dim=128):
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

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


def make_normalizer(x, eps=1e-6):
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps)
    return Normalizer(mean, std)


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(os.path.join(data_dir, "tracer_dataset_v1_*.npz")))
    if not files:
        raise FileNotFoundError(f"No tracer_dataset_v1_*.npz found in {data_dir}")

    xs = []
    ys = []

    for path in files:
        d = np.load(path)

        relative_goal = d["relative_goal"]
        context = d["context"]
        latent = d["latent"]
        objective_weights = d["objective_weights"]
        mpc = d["mpc_reference"]
        valid = d["valid_mask"] > 0.5

        beta_sum = objective_weights.sum(axis=1)
        valid = valid & np.isfinite(beta_sum) & (beta_sum > 0.5)

        x = np.concatenate(
            [
                relative_goal[valid],
                context[valid],
                latent[valid],
                objective_weights[valid],
            ],
            axis=1,
        )

        # mpc_reference = [counter, vx, yaw_rate, body_height, clearance, enable]
        y = mpc[valid][:, 1:6]

        finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x = x[finite]
        y = y[finite]

        xs.append(x)
        ys.append(y)

        print(f"[load] {os.path.basename(path)} x={x.shape} y={y.shape}")

    X = np.concatenate(xs, axis=0).astype(np.float32)
    Y = np.concatenate(ys, axis=0).astype(np.float32)

    print(f"[dataset] X={X.shape}, Y={Y.shape}")

    # Quick mode distribution check.
    beta = X[:, -3:]
    print("[beta stats]")
    print("  mean:", beta.mean(axis=0))
    print("  min: ", beta.min(axis=0))
    print("  max: ", beta.max(axis=0))

    print("[target stats]")
    print("  vx min/mean/max:", Y[:, 0].min(), Y[:, 0].mean(), Y[:, 0].max())
    print("  h  min/mean/max:", Y[:, 2].min(), Y[:, 2].mean(), Y[:, 2].max())
    print("  cl min/mean/max:", Y[:, 3].min(), Y[:, 3].mean(), Y[:, 3].max())

    return X, Y


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)

    X, Y = load_dataset(args.data_dir)

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

    model = HighLevelPolicyV1MLP(
        input_dim=X.shape[1],
        output_dim=Y.shape[1],
        hidden_dim=args.hidden_dim,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "tracer_high_level_policy_v1_best.pt")
    final_path = os.path.join(args.out_dir, "tracer_high_level_policy_v1_final.pt")
    ts_path = os.path.join(args.out_dir, "tracer_high_level_policy_v1.ts")

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
                    "input_layout": "relative_goal[4]+context[16]+latent[16]+objective_weights[3]",
                    "output_layout": "vx,yaw_rate,body_height,clearance,enable",
                },
                best_path,
            )

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"[epoch {epoch:04d}] "
                f"train={train_loss:.8f} val={val_loss:.8f} best={best_val:.8f}"
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
            "input_layout": "relative_goal[4]+context[16]+latent[16]+objective_weights[3]",
            "output_layout": "vx,yaw_rate,body_height,clearance,enable",
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
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--print-every", type=int, default=10)

    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
