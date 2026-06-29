#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class RAMGRUV2(nn.Module):
    def __init__(self, input_dim: int = 57, hidden_dim: int = 64, out_dim: int = 5):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        _, h = self.gru(x)
        z = h[-1]
        return self.head(z)


def split_indices(n: int, seed: int = 7, train_frac: float = 0.8):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * train_frac))
    return idx[:n_train], idx[n_train:]


def binarize_labels(y_raw: np.ndarray) -> np.ndarray:
    y = np.zeros_like(y_raw, dtype=np.float32)

    # Label order:
    # 0 future_fallen
    # 1 future_recovery_needed
    # 2 future_invalid_runtime
    # 3 future_bad_locomotion
    # 4 future_good_locomotion
    y[:, 0] = (y_raw[:, 0] >= 0.50).astype(np.float32)
    y[:, 1] = (y_raw[:, 1] >= 0.50).astype(np.float32)
    y[:, 2] = (y_raw[:, 2] >= 0.50).astype(np.float32)
    y[:, 3] = (y_raw[:, 3] >= 0.50).astype(np.float32)
    y[:, 4] = (y_raw[:, 4] >= 0.50).astype(np.float32)
    return y


def masked_bce_with_logits(logits, y, active_mask, pos_weight):
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        y,
        reduction="none",
        pos_weight=pos_weight,
    )
    loss = loss * active_mask.reshape(1, -1)
    denom = torch.clamp(active_mask.sum(), min=1.0)
    return loss.sum(dim=1).mean() / denom


def metrics(logits, y, active_mask):
    with torch.no_grad():
        p = torch.sigmoid(logits)
        pred = (p >= 0.5).float()

        acc = (pred == y).float().mean(dim=0)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            y,
            reduction="none",
        ).mean(dim=0)

        prob_mean = p.mean(dim=0)
        prob_std = p.std(dim=0)

    return {
        "acc": acc.cpu().numpy(),
        "bce": bce.cpu().numpy(),
        "prob_mean": prob_mean.cpu().numpy(),
        "prob_std": prob_std.cpu().numpy(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/training/tracer_ram_window_dataset_v2.npz")
    ap.add_argument("--artifact-dir", default="artifacts/tracer_ram_v2")
    ap.add_argument("--tracked-model", default="configs/learned_models/tracer_ram_v2_model.pt")
    ap.add_argument("--summary-json", default="reports/tracer_ram_v2_train_summary.json")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.dataset, allow_pickle=True)
    X = data["windows"].astype(np.float32)
    Y_raw = data["labels"].astype(np.float32)

    # Robust finite sanitation.
    # Some runtime age/debug features can be inf/nan when a stream is stale.
    # Without this, mean/std normalization can become nan and the GRU collapses immediately.
    input_nonfinite_count = int((~np.isfinite(X)).sum())
    label_nonfinite_count = int((~np.isfinite(Y_raw)).sum())

    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    Y_raw = np.nan_to_num(Y_raw, nan=0.0, posinf=1.0, neginf=0.0)

    # Clip extreme scalar features while preserving normal pose/age ranges.
    X = np.clip(X, -1000.0, 1000.0)

    Y = binarize_labels(Y_raw)
    label_names = [str(x) for x in data["label_names"]]

    n, T, D = X.shape
    train_idx, val_idx = split_indices(n, seed=args.seed)

    mean = X[train_idx].reshape(-1, D).mean(axis=0)
    std = X[train_idx].reshape(-1, D).std(axis=0)

    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std[std < 1e-6] = 1.0

    Xs = (X - mean.reshape(1, 1, D)) / std.reshape(1, 1, D)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=10.0, neginf=-10.0)
    Xs = np.clip(Xs, -20.0, 20.0)

    x_train = torch.tensor(Xs[train_idx], dtype=torch.float32)
    y_train = torch.tensor(Y[train_idx], dtype=torch.float32)
    x_val = torch.tensor(Xs[val_idx], dtype=torch.float32)
    y_val = torch.tensor(Y[val_idx], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    pos = y_train.sum(dim=0)
    neg = y_train.shape[0] - pos

    # Active head only if both positive and negative examples exist.
    active_mask = ((pos > 0) & (neg > 0)).float()

    # recovery/invalid heads usually have no signal in v2, so they will be inactive.
    pos_weight = neg / torch.clamp(pos, min=1.0)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=20.0)

    print("[TRACER] active heads")
    for i, name in enumerate(label_names):
        print(
            f"  {name:28s} "
            f"pos={float(pos[i]):7.1f} neg={float(neg[i]):7.1f} "
            f"active={int(active_mask[i].item())} "
            f"pos_weight={float(pos_weight[i]):.3f}"
        )

    model = RAMGRUV2(input_dim=D, hidden_dim=args.hidden_dim, out_dim=Y.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(args.epochs):
        model.train()

        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = masked_bce_with_logits(logits, yb, active_mask, pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(x_train)
            val_logits = model(x_val)
            train_loss = float(masked_bce_with_logits(train_logits, y_train, active_mask, pos_weight).item())
            val_loss = float(masked_bce_with_logits(val_logits, y_val, active_mask, pos_weight).item())
            train_m = metrics(train_logits, y_train, active_mask)
            val_m = metrics(val_logits, y_val, active_mask)

        active_np = active_mask.cpu().numpy().astype(bool)
        train_acc_mean = float(train_m["acc"][active_np].mean()) if active_np.any() else 0.0
        val_acc_mean = float(val_m["acc"][active_np].mean()) if active_np.any() else 0.0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc_mean_active": train_acc_mean,
            "val_acc_mean_active": val_acc_mean,
        })

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }

        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(
                f"[TRACER] epoch={epoch:04d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc_active={val_acc_mean:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    model.eval()
    with torch.no_grad():
        train_logits = model(x_train)
        val_logits = model(x_val)
        train_m = metrics(train_logits, y_train, active_mask)
        val_m = metrics(val_logits, y_val, active_mask)
        val_prob = torch.sigmoid(val_logits).cpu().numpy()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "model.pt"
    tracked_model_path = Path(args.tracked_model)
    tracked_model_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_state_dict": model.state_dict(),
        "input_dim": D,
        "window": T,
        "hidden_dim": args.hidden_dim,
        "label_names": label_names,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "active_head_mask": active_mask.cpu().numpy().astype(float).tolist(),
        "dataset": args.dataset,
        "label_mode": "binary_thresholded_v2b",
    }

    torch.save(artifact, model_path)
    torch.save(artifact, tracked_model_path)

    summary_path = Path(args.summary_json)

    summary = {
        "dataset": args.dataset,
        "artifact_dir": str(artifact_dir),
        "model_path": str(model_path),
        "tracked_model": str(tracked_model_path),
        "summary_json": str(summary_path),
        "n_windows": int(n),
        "train_windows": int(len(train_idx)),
        "val_windows": int(len(val_idx)),
        "window": int(T),
        "input_dim": int(D),
        "label_names": label_names,
        "input_nonfinite_count": input_nonfinite_count,
        "label_nonfinite_count": label_nonfinite_count,
        "raw_label_means": {name: float(Y_raw[:, i].mean()) for i, name in enumerate(label_names)},
        "binary_label_means": {name: float(Y[:, i].mean()) for i, name in enumerate(label_names)},
        "active_head_mask": {name: float(active_mask[i]) for i, name in enumerate(label_names)},
        "best_val_loss": float(best_val),
        "best_epoch": int(best_state["epoch"]) if best_state else None,
        "train_acc": {name: float(train_m["acc"][i]) for i, name in enumerate(label_names)},
        "val_acc": {name: float(val_m["acc"][i]) for i, name in enumerate(label_names)},
        "train_bce": {name: float(train_m["bce"][i]) for i, name in enumerate(label_names)},
        "val_bce": {name: float(val_m["bce"][i]) for i, name in enumerate(label_names)},
        "val_prob_mean": {name: float(val_m["prob_mean"][i]) for i, name in enumerate(label_names)},
        "val_prob_std": {name: float(val_m["prob_std"][i]) for i, name in enumerate(label_names)},
        "pos_weight": {name: float(pos_weight[i]) for i, name in enumerate(label_names)},
        "history_tail": history[-10:],
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print("[TRACER] RAM v2b training summary")
    print(json.dumps(summary, indent=2))

    print("\n[TRACER] validation probabilities sample")
    for i in range(min(10, len(val_idx))):
        probs = {name: float(val_prob[i, j]) for j, name in enumerate(label_names)}
        target = {name: float(Y[val_idx[i], j]) for j, name in enumerate(label_names)}
        print(f"idx={int(val_idx[i])} target={target} pred={probs}")


if __name__ == "__main__":
    main()
