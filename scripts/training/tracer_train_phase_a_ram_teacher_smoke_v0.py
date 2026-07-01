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

TARGET_KEYS = [
    "ram_vertical_sink_target",
    "ram_progress_mismatch_target",
    "ram_orientation_risk_target",
    "ram_low_height_risk_target",
    "ram_recovery_needed_target",
    "ram_uncertainty_proxy_target",
]


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def terrain_onehot(terrain: str, vocab: list[str]) -> list[float]:
    return [1.0 if terrain == t else 0.0 for t in vocab]


def row_to_x(row: dict[str, Any], terrain_vocab: list[str]) -> list[float]:
    terrain = str(row.get("terrain", "unknown"))

    return [
        *terrain_onehot(terrain, terrain_vocab),
        to_float(row.get("beta_motion")),
        to_float(row.get("beta_stability")),
        to_float(row.get("beta_energy")),
        to_float(row.get("a_hl_vx")),
        to_float(row.get("a_hl_yaw_rate")),
        to_float(row.get("a_hl_body_height")),
        to_float(row.get("a_hl_swing_clearance")),
    ]


def row_to_y(row: dict[str, Any]) -> list[float]:
    return [to_float(row.get(k)) for k in TARGET_KEYS]


def row_weight(row: dict[str, Any]) -> float:
    label = str(row.get("manifest_label", ""))
    terrain = str(row.get("terrain", ""))

    w = 1.0

    if label == "positive_teacher":
        w *= 1.25
    elif label == "borderline_teacher":
        w *= 1.10
    elif label == "risk_recovery":
        w *= 1.20

    # Slippery risk/borderline is important for uncertainty/recovery heads.
    if terrain == "slippery_mild_flat":
        w *= 1.35

    return w


class RamTeacherSmoke(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.Tanh(),
            nn.Linear(96, 96),
            nn.Tanh(),
            nn.Linear(96, 64),
            nn.Tanh(),
            nn.Linear(64, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest-csv",
        default="data/phase_a_training_manifest_v0/phase_a_training_manifest_v0.csv",
    )
    ap.add_argument("--out-dir", default="artifacts/phase_a_ram_teacher_smoke_v0")
    ap.add_argument("--terrain-vocab", default=",".join(DEFAULT_TERRAINS))
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    terrain_vocab = [x.strip() for x in args.terrain_vocab.split(",") if x.strip()]
    rows = read_csv(Path(args.manifest_csv))
    if not rows:
        raise RuntimeError(f"No rows found: {args.manifest_csv}")

    xs = torch.tensor([row_to_x(r, terrain_vocab) for r in rows], dtype=torch.float32)
    ys = torch.tensor([row_to_y(r) for r in rows], dtype=torch.float32)
    ws = torch.tensor([row_weight(r) for r in rows], dtype=torch.float32).view(-1, 1)

    model = RamTeacherSmoke(in_dim=xs.shape[1], out_dim=ys.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        pred = model(xs)
        loss = (((pred - ys) ** 2) * ws).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep == 1 or ep % 300 == 0 or ep == args.epochs:
            with torch.no_grad():
                mae_each = (pred - ys).abs().mean(dim=0)
                mae = mae_each.mean().item()
            mae_txt = " ".join(f"{k}={v.item():.4f}" for k, v in zip(TARGET_KEYS, mae_each))
            print(f"epoch={ep:04d} loss={loss.item():.8f} mae={mae:.5f} {mae_txt}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "phase_a_ram_teacher_smoke_v0.pt"
    meta_path = out_dir / "phase_a_ram_teacher_smoke_v0_meta.json"

    input_schema = [
        *[f"terrain_onehot:{t}" for t in terrain_vocab],
        "beta_motion",
        "beta_stability",
        "beta_energy",
        "a_hl_vx",
        "a_hl_yaw_rate",
        "a_hl_body_height",
        "a_hl_swing_clearance",
    ]

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(xs.shape[1]),
            "output_dim": int(ys.shape[1]),
            "terrain_vocab": terrain_vocab,
            "input_schema": input_schema,
            "target_schema": TARGET_KEYS,
            "note": (
                "Phase-A RAM teacher-student smoke model. This predicts teacher "
                "risk/mismatch targets derived from rollout manifest. It is a scaffold, "
                "not the final recurrent/proprioceptive RAM."
            ),
        },
        model_path,
    )

    with torch.no_grad():
        pred = model(xs).detach()

    summary = {
        "manifest_csv": args.manifest_csv,
        "num_rows": len(rows),
        "terrain_vocab": terrain_vocab,
        "input_schema": input_schema,
        "target_schema": TARGET_KEYS,
        "x_mean": xs.mean(dim=0).tolist(),
        "y_mean": ys.mean(dim=0).tolist(),
        "mae": (pred - ys).abs().mean(dim=0).tolist(),
        "model_path": str(model_path),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"[TRACER] wrote model: {model_path}")
    print(f"[TRACER] wrote meta:  {meta_path}")

    print()
    print("===== sample predictions =====")
    order = torch.argsort(ys[:, TARGET_KEYS.index("ram_uncertainty_proxy_target")], descending=True).tolist()
    for i in order[:20]:
        r = rows[i]
        p = pred[i]
        y = ys[i]
        print(
            f"terrain={r.get('terrain',''):20s} "
            f"label={r.get('manifest_label',''):18s} "
            f"profile={r.get('profile',''):18s} "
            f"preset={r.get('preset','')[:42]:42s}"
        )
        print("  target:", " ".join(f"{k}={y[j].item():.3f}" for j, k in enumerate(TARGET_KEYS)))
        print("  pred:  ", " ".join(f"{k}={p[j].item():.3f}" for j, k in enumerate(TARGET_KEYS)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
