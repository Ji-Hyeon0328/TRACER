#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from tracer_core.highlevel.meta_gait_dataset import (
    FEATURE_NAMES,
    TARGET_NAMES,
    build_meta_gait_samples_from_policy_json,
    write_jsonl,
)


class MetaGaitMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_jsonl_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise RuntimeError(f"Empty dataset: {path}")

    feature_names = rows[0].get("feature_names", list(FEATURE_NAMES))
    target_names = rows[0].get("target_names", list(TARGET_NAMES))

    xs = torch.tensor([r["feature_vector"] for r in rows], dtype=torch.float32)
    ys = torch.tensor([r["target_vector"] for r in rows], dtype=torch.float32)

    if xs.ndim != 2 or ys.ndim != 2:
        raise RuntimeError(f"Bad dataset shape: x={tuple(xs.shape)} y={tuple(ys.shape)}")

    return xs, ys, feature_names, target_names, rows


def standardize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (x - mean) / std, mean, std


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset).expanduser()

    if not dataset_path.exists():
        if not args.policy_json:
            raise FileNotFoundError(
                f"Dataset does not exist: {dataset_path}. "
                f"Pass --policy-json to build it automatically."
            )
        samples = build_meta_gait_samples_from_policy_json(args.policy_json)
        write_jsonl(samples, dataset_path)
        print(f"[TRACER] built dataset: {dataset_path} samples={len(samples)}")

    x_raw, y_raw, feature_names, target_names, rows = load_jsonl_dataset(dataset_path)
    n = x_raw.shape[0]

    x, x_mean, x_std = standardize(x_raw)
    y, y_mean, y_std = standardize(y_raw)

    indices = list(range(n))
    random.shuffle(indices)

    val_n = int(round(n * args.val_fraction))
    if n >= 5:
        val_n = max(1, min(val_n, n - 1))
    else:
        val_n = 0

    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx] if val_idx else None
    y_val = y[val_idx] if val_idx else None

    model = MetaGaitMLP(
        input_dim=x.shape[1],
        output_dim=y.shape[1],
        hidden_dim=args.hidden_dim,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            msg = f"[epoch {epoch:04d}] train_mse={loss.item():.8f}"
            if x_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    val_loss = loss_fn(model(x_val), y_val).item()
                msg += f" val_mse={val_loss:.8f}"
            print(msg)

    model.eval()
    with torch.no_grad():
        y_hat_norm = model(x)
        y_hat = y_hat_norm * y_std + y_mean
        abs_err = (y_hat - y_raw).abs()
        mae = abs_err.mean(dim=0)

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pt"
    stats_path = out_dir / "stats.json"
    preview_path = out_dir / "preview.json"

    bundle = {
        "model_state_dict": model.state_dict(),
        "input_dim": int(x.shape[1]),
        "output_dim": int(y.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "feature_names": list(feature_names),
        "target_names": list(target_names),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }
    torch.save(bundle, model_path)

    stats = {
        "dataset": str(dataset_path),
        "num_samples": int(n),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "feature_names": list(feature_names),
        "target_names": list(target_names),
        "target_mae": {name: float(mae[i]) for i, name in enumerate(target_names)},
        "model_path": str(model_path),
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))

    preview = []
    for i in range(min(args.preview, n)):
        preview.append(
            {
                "terrain_key": rows[i].get("terrain_key", f"sample_{i}"),
                "target": {
                    name: float(y_raw[i, j])
                    for j, name in enumerate(target_names)
                },
                "pred": {
                    name: float(y_hat[i, j])
                    for j, name in enumerate(target_names)
                },
            }
        )
    preview_path.write_text(json.dumps(preview, indent=2, sort_keys=True))

    print("\n[TRACER] training complete")
    print(f"[TRACER] dataset: {dataset_path}")
    print(f"[TRACER] samples: {n}")
    print(f"[TRACER] model:   {model_path}")
    print(f"[TRACER] stats:   {stats_path}")
    print(f"[TRACER] preview: {preview_path}")
    print("[TRACER] target MAE:")
    for name in target_names:
        print(f"  {name:20s} {stats['target_mae'][name]:.8f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/training/meta_gait_policy_dataset_v0.jsonl",
        help="Input jsonl dataset. If missing, --policy-json can be used to build it.",
    )
    parser.add_argument(
        "--policy-json",
        default="configs/highlevel_policy/tracer_fusion_policy_slippery_micro_brake_v3.json",
        help="Fusion policy json used to auto-build dataset if --dataset is missing.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/meta_gait_policy_v0",
        help="Output directory for model/stats.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--preview", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
