#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


DEFAULT_TERRAINS = [
    "flat_normal",
    "sponge_firm_flat",
    "slippery_mild_flat",
]


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def parse_beta(s: Any) -> list[float]:
    if isinstance(s, list):
        out = [float(x) for x in s]
    else:
        try:
            out = [float(x) for x in json.loads(str(s))]
        except Exception:
            out = [0.34, 0.33, 0.33]

    if len(out) < 3:
        out = (out + [0.0, 0.0, 0.0])[:3]
    return out[:3]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def terrain_onehot(terrain: str, vocab: list[str]) -> list[float]:
    return [1.0 if terrain == t else 0.0 for t in vocab]


def row_to_x(row: dict[str, Any], terrain_vocab: list[str]) -> list[float]:
    beta = parse_beta(row.get("beta", ""))
    terrain = str(row.get("terrain", "unknown"))

    return [
        beta[0],
        beta[1],
        beta[2],
        *terrain_onehot(terrain, terrain_vocab),
        to_float(row.get("ref_vx")),
        to_float(row.get("ref_yaw_rate")),
        to_float(row.get("ref_body_height")),
        to_float(row.get("ref_swing_clearance")),
    ]


def row_to_y(row: dict[str, Any]) -> float:
    if "R_train_score" in row:
        return clamp(to_float(row.get("R_train_score")))
    if "R_profile_v2_gated_mean" in row:
        return clamp(to_float(row.get("R_profile_v2_gated_mean")))
    return clamp(to_float(row.get("R_gated_mean")))


def row_weight(row: dict[str, Any]) -> float:
    label = str(row.get("bank_label", ""))
    terrain = str(row.get("terrain", ""))

    # Do not throw away risk negatives. They are essential for RAM/ranker.
    # But give positives/borderlines slightly more weight so the best action
    # surface is not drowned by many failure samples.
    w = 1.0
    if label == "positive_teacher":
        w *= 1.5
    elif label == "borderline":
        w *= 1.25
    elif label == "risk_negative":
        w *= 1.0

    # Slippery has many risk rows and few borderline rows in current data.
    # Keep it visible during early training.
    if terrain == "slippery_mild_flat" and label == "borderline":
        w *= 1.5

    return float(w)


class TerrainAwareRanker(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.Tanh(),
            nn.Linear(96, 96),
            nn.Tanh(),
            nn.Linear(96, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate-csv",
        default="data/phase_a_terrain_aware_bank_v0/phase_a_terrain_aware_candidates_v0.csv",
    )
    ap.add_argument("--out-dir", default="artifacts/phase_a_terrain_aware_ranker_v0")
    ap.add_argument("--terrain-vocab", default=",".join(DEFAULT_TERRAINS))
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    terrain_vocab = [x.strip() for x in args.terrain_vocab.split(",") if x.strip()]
    rows = read_csv(Path(args.candidate_csv))
    if not rows:
        raise RuntimeError(f"No rows found: {args.candidate_csv}")

    xs = torch.tensor([row_to_x(r, terrain_vocab) for r in rows], dtype=torch.float32)
    ys = torch.tensor([row_to_y(r) for r in rows], dtype=torch.float32)
    ws = torch.tensor([row_weight(r) for r in rows], dtype=torch.float32)

    model = TerrainAwareRanker(in_dim=xs.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        pred = model(xs)
        loss = ((pred - ys) ** 2 * ws).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep == 1 or ep % 300 == 0 or ep == args.epochs:
            with torch.no_grad():
                mae = (pred - ys).abs().mean().item()
                wmae = ((pred - ys).abs() * ws).sum().item() / ws.sum().item()
            print(f"epoch={ep:04d} loss={loss.item():.8f} mae={mae:.5f} wmae={wmae:.5f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "phase_a_terrain_aware_ranker_v0.pt"
    meta_path = out_dir / "phase_a_terrain_aware_ranker_v0_meta.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(xs.shape[1]),
            "terrain_vocab": terrain_vocab,
            "input_schema": [
                "beta_motion",
                "beta_stability",
                "beta_energy",
                *[f"terrain_onehot:{t}" for t in terrain_vocab],
                "candidate_vx",
                "candidate_yaw_rate",
                "candidate_body_height",
                "candidate_swing_clearance",
            ],
            "target_schema": ["R_train_score"],
            "note": (
                "Phase-A terrain-aware candidate ranker. Scores candidate high-level "
                "references conditioned on objective beta and terrain identity. "
                "This is a substrate for TRACER RAM/objective/RL warm-start, not final policy."
            ),
        },
        model_path,
    )

    summary: dict[str, Any] = {
        "candidate_csv": args.candidate_csv,
        "num_rows": len(rows),
        "terrain_vocab": terrain_vocab,
        "input_dim": int(xs.shape[1]),
        "x_mean": xs.mean(dim=0).tolist(),
        "y_mean": float(ys.mean().item()),
        "y_min": float(ys.min().item()),
        "y_max": float(ys.max().item()),
        "model_path": str(model_path),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"[TRACER] wrote model: {model_path}")
    print(f"[TRACER] wrote meta:  {meta_path}")
    print()

    with torch.no_grad():
        pred = model(xs).detach()

    print("===== top predicted by terrain/profile =====")
    terrains = sorted(set(str(r.get("terrain", "unknown")) for r in rows))
    profiles = sorted(set(str(r.get("profile", "unknown")) for r in rows))

    for terrain in terrains:
        print()
        print("=" * 80)
        print(f"terrain={terrain}")
        for profile in profiles:
            idxs = [
                i for i, r in enumerate(rows)
                if str(r.get("terrain", "unknown")) == terrain
                and str(r.get("profile", "unknown")) == profile
            ]
            if not idxs:
                continue

            idxs = sorted(idxs, key=lambda i: float(pred[i].item()), reverse=True)
            i = idxs[0]
            r = rows[i]
            print(
                f"  profile={profile:18s} "
                f"label={r.get('bank_label',''):16s} "
                f"preset={r.get('preset',''):36s} "
                f"true={ys[i].item():.4f} pred={pred[i].item():.4f} "
                f"vx={to_float(r.get('ref_vx')):.3f} "
                f"h={to_float(r.get('ref_body_height')):.3f} "
                f"c={to_float(r.get('ref_swing_clearance')):.3f}"
            )

    print()
    print("===== largest absolute errors =====")
    err_order = torch.argsort((pred - ys).abs(), descending=True).tolist()
    for i in err_order[:25]:
        r = rows[i]
        print(
            f"terrain={str(r.get('terrain','')):20s} "
            f"profile={str(r.get('profile','')):18s} "
            f"label={str(r.get('bank_label','')):16s} "
            f"preset={str(r.get('preset','')):36s} "
            f"true={ys[i].item():.4f} pred={pred[i].item():.4f} "
            f"err={abs(pred[i].item() - ys[i].item()):.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
