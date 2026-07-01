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
    "empirical_recovery_prob",
    "empirical_uncertainty",
    "empirical_stable_prob_lcb95",
    "empirical_zmin_risk",
    "empirical_below22_risk",
    "empirical_teacher_weight",
]


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


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
    recovery = to_float(row.get("empirical_recovery_prob"))
    uncertainty = to_float(row.get("empirical_uncertainty"))
    lcb = to_float(row.get("empirical_stable_prob_lcb95"))

    zmin = to_float(row.get("empirical_zmin_mean"))
    below22 = to_float(row.get("empirical_below22_mean"))

    # 0 means safe height, 1 means severe low-height risk.
    zmin_risk = clamp01((0.24 - zmin) / 0.16)
    below22_risk = clamp01(below22)

    teacher_w = to_float(row.get("empirical_teacher_weight"))

    return [
        clamp01(recovery),
        clamp01(uncertainty),
        clamp01(lcb),
        zmin_risk,
        below22_risk,
        clamp01(teacher_w),
    ]


def row_weight(row: dict[str, Any]) -> float:
    label = str(row.get("empirical_outcome_label", ""))
    recovery = to_float(row.get("empirical_recovery_prob"))
    uncertainty = to_float(row.get("empirical_uncertainty"))
    teacher_w = to_float(row.get("empirical_teacher_weight"))

    w = 0.5 + teacher_w

    # RAM must learn risk and uncertainty strongly, not only good teachers.
    w += 0.6 * recovery
    w += 0.4 * uncertainty

    if label == "risk_negative":
        w *= 1.25
    elif label == "borderline_high_variance":
        w *= 1.15
    elif label == "probabilistic_positive":
        w *= 1.05

    return max(0.1, float(w))


class RamEmpiricalV1(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--empirical-manifest-csv",
        default="data/phase_a_empirical_manifest_v1/phase_a_empirical_manifest_v1.csv",
    )
    ap.add_argument("--out-dir", default="artifacts/phase_a_ram_empirical_v1")
    ap.add_argument("--terrain-vocab", default=",".join(DEFAULT_TERRAINS))
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    terrain_vocab = [x.strip() for x in args.terrain_vocab.split(",") if x.strip()]
    rows_all = read_csv(Path(args.empirical_manifest_csv))

    rows = [r for r in rows_all if int(to_float(r.get("empirical_available"))) == 1]
    if not rows:
        raise RuntimeError("No empirical rows available")

    xs = torch.tensor([row_to_x(r, terrain_vocab) for r in rows], dtype=torch.float32)
    ys = torch.tensor([row_to_y(r) for r in rows], dtype=torch.float32)
    ws = torch.tensor([row_weight(r) for r in rows], dtype=torch.float32).view(-1, 1)

    model = RamEmpiricalV1(in_dim=xs.shape[1], out_dim=ys.shape[1])
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

    model_path = out_dir / "phase_a_ram_empirical_v1.pt"
    meta_path = out_dir / "phase_a_ram_empirical_v1_meta.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(xs.shape[1]),
            "output_dim": int(ys.shape[1]),
            "terrain_vocab": terrain_vocab,
            "input_schema": input_schema,
            "target_schema": TARGET_KEYS,
            "note": (
                "Phase-A empirical RAM v1. Predicts aggregate recovery probability, "
                "uncertainty, stable LCB, and height risk from terrain/objective/action."
            ),
        },
        model_path,
    )

    with torch.no_grad():
        pred = model(xs).detach()
        mae_each = (pred - ys).abs().mean(dim=0)

    summary = {
        "empirical_manifest_csv": args.empirical_manifest_csv,
        "num_rows": len(rows),
        "terrain_vocab": terrain_vocab,
        "input_schema": input_schema,
        "target_schema": TARGET_KEYS,
        "mae": {k: float(v) for k, v in zip(TARGET_KEYS, mae_each)},
        "model_path": str(model_path),
    }

    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"[TRACER] wrote model: {model_path}")
    print(f"[TRACER] wrote meta:  {meta_path}")

    print()
    print("===== high-risk / high-uncertainty samples =====")
    with torch.no_grad():
        order = torch.argsort(ys[:, 0] + ys[:, 1], descending=True).tolist()

    for i in order[:25]:
        r = rows[i]
        y = ys[i]
        p = pred[i]
        print(
            f"terrain={r.get('terrain',''):20s} "
            f"label={r.get('empirical_outcome_label',''):25s} "
            f"profile={r.get('profile',''):18s} "
            f"a=[{to_float(r.get('a_hl_vx')):.3f},"
            f"{to_float(r.get('a_hl_body_height')):.3f},"
            f"{to_float(r.get('a_hl_swing_clearance')):.3f}]"
        )
        print("  target:", " ".join(f"{k}={y[j].item():.3f}" for j, k in enumerate(TARGET_KEYS)))
        print("  pred:  ", " ".join(f"{k}={p[j].item():.3f}" for j, k in enumerate(TARGET_KEYS)))

    print()
    print("===== best teacher-weight samples =====")
    order2 = torch.argsort(ys[:, TARGET_KEYS.index("empirical_teacher_weight")], descending=True).tolist()
    for i in order2[:20]:
        r = rows[i]
        y = ys[i]
        p = pred[i]
        print(
            f"terrain={r.get('terrain',''):20s} "
            f"label={r.get('empirical_outcome_label',''):25s} "
            f"profile={r.get('profile',''):18s} "
            f"a=[{to_float(r.get('a_hl_vx')):.3f},"
            f"{to_float(r.get('a_hl_body_height')):.3f},"
            f"{to_float(r.get('a_hl_swing_clearance')):.3f}] "
            f"teacher_w={y[TARGET_KEYS.index('empirical_teacher_weight')].item():.3f} "
            f"pred_w={p[TARGET_KEYS.index('empirical_teacher_weight')].item():.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
