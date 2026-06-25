#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Important:
# Do NOT include trajectory_cost_v0, R_tracer_slide_v0, R_motion,
# R_stability, R_energy_proxy, or R_aux.
# This model should learn the preference relation from raw rollout/command/beta features.
FEATURE_KEYS = [
    "command_vx",
    "command_yaw",
    "command_body_height",
    "command_clearance",
    "command_enable",

    "dx",
    "dy",
    "min_z",
    "max_abs_roll",
    "max_abs_pitch",

    "beta_motion",
    "beta_stability",
    "beta_energy",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
    except Exception:
        pass
    return default


def feature_from_prefix(row: dict[str, Any], prefix: str) -> list[float]:
    return [as_float(row.get(f"{prefix}_{k}", 0.0)) for k in FEATURE_KEYS]


def build_dataset(
    rows: list[dict[str, Any]],
    *,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xw, xl, weights, used_rows = [], [], [], []

    for r in rows:
        conf = as_float(r.get("confidence", 1.0), 1.0)
        if conf < min_confidence:
            continue

        xw.append(feature_from_prefix(r, "winner"))
        xl.append(feature_from_prefix(r, "loser"))
        weights.append(conf)
        used_rows.append(r)

    if not xw:
        raise RuntimeError("No preference pairs after filtering.")

    return (
        np.asarray(xw, dtype=np.float32),
        np.asarray(xl, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        used_rows,
    )


class RewardNet(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.maximum(std, 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pairs-jsonl",
        default="data/preference_datasets/tracer_slide_reward_preference_pairs_v0_clean.jsonl",
    )
    ap.add_argument("--out-dir", default="artifacts/preference_reward_slide_v0")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--min-confidence", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_jsonl(Path(args.pairs_jsonl))
    xw, xl, weights, used_rows = build_dataset(
        rows,
        min_confidence=args.min_confidence,
    )

    x_all = np.concatenate([xw, xl], axis=0)
    mean = x_all.mean(axis=0)
    std = x_all.std(axis=0)

    xw_s = standardize(xw, mean, std)
    xl_s = standardize(xl, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xw_t = torch.tensor(xw_s, dtype=torch.float32, device=device)
    xl_t = torch.tensor(xl_s, dtype=torch.float32, device=device)
    w_t = torch.tensor(weights, dtype=torch.float32, device=device)

    model = RewardNet(dim=xw.shape[1], hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(args.epochs):
        sw = model(xw_t)
        sl = model(xl_t)
        logits = sw - sl

        # Bradley-Terry preference loss:
        # winner should have higher scalar reward score.
        loss_raw = F.softplus(-logits)
        loss = (loss_raw * w_t).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            acc = (logits > 0.0).float().mean().item()
            margin = logits.mean().item()

        if epoch == 0 or (epoch + 1) % 100 == 0 or epoch == args.epochs - 1:
            item = {
                "epoch": epoch + 1,
                "loss": float(loss.item()),
                "pair_acc": float(acc),
                "mean_margin": float(margin),
            }
            history.append(item)
            print(
                f"epoch={epoch+1:04d} "
                f"loss={loss.item():.6f} "
                f"pair_acc={acc:.3f} "
                f"mean_margin={margin:.3f}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "feature_keys": FEATURE_KEYS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "hidden": args.hidden,
        "num_pairs": int(len(xw)),
        "min_confidence": args.min_confidence,
        "pairs_jsonl": args.pairs_jsonl,
        "note": (
            "Slide reward preference model v0. "
            "Does not use trajectory_cost_v0 or explicit slide reward components as input."
        ),
    }

    torch.save(ckpt, out_dir / "model.pt")

    stats = {
        "pairs_jsonl": args.pairs_jsonl,
        "num_input_pairs": len(rows),
        "num_used_pairs": int(len(xw)),
        "feature_keys": FEATURE_KEYS,
        "excluded_shortcuts": [
            "trajectory_cost_v0",
            "R_tracer_slide_v0",
            "R_motion",
            "R_stability",
            "R_energy_proxy",
            "R_aux",
        ],
        "history": history,
        "device": str(device),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    with torch.no_grad():
        sw = model(xw_t).detach().cpu().numpy()
        sl = model(xl_t).detach().cpu().numpy()

    preview = []
    for i, r in enumerate(used_rows[:20]):
        preview.append({
            "pair_id": r.get("pair_id"),
            "reason": r.get("reason"),
            "pair_type": r.get("pair_type"),
            "confidence": as_float(r.get("confidence", 1.0), 1.0),
            "reward_gap_label": as_float(r.get("reward_gap", 0.0), 0.0),
            "winner_style": r.get("winner_style"),
            "loser_style": r.get("loser_style"),
            "winner_score": float(sw[i]),
            "loser_score": float(sl[i]),
            "score_margin": float(sw[i] - sl[i]),
        })

    (out_dir / "preview.json").write_text(json.dumps(preview, indent=2) + "\n")

    print()
    print(f"[TRACER] saved model:   {out_dir / 'model.pt'}")
    print(f"[TRACER] saved stats:   {out_dir / 'stats.json'}")
    print(f"[TRACER] saved preview: {out_dir / 'preview.json'}")


if __name__ == "__main__":
    main()
