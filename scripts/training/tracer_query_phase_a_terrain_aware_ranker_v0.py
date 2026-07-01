#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


PROFILE_BETA = {
    "balanced":          [0.34, 0.33, 0.33],
    "motion":            [0.65, 0.20, 0.15],
    "stability":         [0.20, 0.65, 0.15],
    "energy":            [0.20, 0.20, 0.60],
    "motion_extreme":    [0.85, 0.10, 0.05],
    "stability_extreme": [0.05, 0.90, 0.05],
    "energy_extreme":    [0.05, 0.10, 0.85],
}


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


def make_x(beta: list[float], terrain: str, row: dict[str, Any], terrain_vocab: list[str]) -> list[float]:
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


def unique_candidates(rows: list[dict[str, Any]], terrain: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()

    for r in rows:
        if str(r.get("terrain", "")) != terrain:
            continue

        key = (
            terrain,
            str(r.get("preset", "")),
            round(to_float(r.get("ref_vx")), 6),
            round(to_float(r.get("ref_yaw_rate")), 6),
            round(to_float(r.get("ref_body_height")), 6),
            round(to_float(r.get("ref_swing_clearance")), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/phase_a_terrain_aware_ranker_v0/phase_a_terrain_aware_ranker_v0.pt")
    ap.add_argument("--candidate-csv", default="data/phase_a_terrain_aware_bank_v0/phase_a_terrain_aware_candidates_v0.csv")
    ap.add_argument("--terrain", required=True)
    ap.add_argument("--profile", default="balanced", choices=sorted(PROFILE_BETA))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out-config", default="")
    ap.add_argument("--preset-name", default="")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    terrain_vocab = list(ckpt.get("terrain_vocab", []))
    input_dim = int(ckpt.get("input_dim"))

    model = TerrainAwareRanker(in_dim=input_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = read_csv(Path(args.candidate_csv))
    candidates = unique_candidates(rows, args.terrain)
    if not candidates:
        raise RuntimeError(f"No candidates for terrain={args.terrain} in {args.candidate_csv}")

    beta = PROFILE_BETA[args.profile]
    xs = torch.tensor([make_x(beta, args.terrain, r, terrain_vocab) for r in candidates], dtype=torch.float32)

    with torch.no_grad():
        scores = model(xs).detach().tolist()

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    print(f"[TRACER] terrain={args.terrain}")
    print(f"[TRACER] profile={args.profile} beta={beta}")
    print(f"[TRACER] candidates={len(candidates)}")
    print()

    print("===== ranked candidates =====")
    for i, (r, s) in enumerate(ranked[: args.top_k], start=1):
        print(
            f"{i:02d}. score={s:.4f} "
            f"label={str(r.get('bank_label','')):16s} "
            f"preset={str(r.get('preset','')):36s} "
            f"vx={to_float(r.get('ref_vx')):.3f} "
            f"yaw={to_float(r.get('ref_yaw_rate')):.3f} "
            f"h={to_float(r.get('ref_body_height')):.3f} "
            f"c={to_float(r.get('ref_swing_clearance')):.3f} "
            f"train_score={to_float(r.get('R_train_score')):.4f}"
        )

    best, best_score = ranked[0]
    vx = to_float(best.get("ref_vx"))
    yaw = to_float(best.get("ref_yaw_rate"))
    h = to_float(best.get("ref_body_height"))
    c = to_float(best.get("ref_swing_clearance"))

    result = {
        "terrain": args.terrain,
        "profile": args.profile,
        "beta": beta,
        "selected": {
            "preset": best.get("preset", ""),
            "bank_label": best.get("bank_label", ""),
            "score": best_score,
            "vx": vx,
            "yaw_rate": yaw,
            "body_height": h,
            "swing_clearance": c,
            "enable": 1.0,
        },
        "note": "Selected by Phase-A terrain-aware candidate ranker.",
    }

    print()
    print(json.dumps(result, indent=2))

    if args.out_config:
        preset_name = args.preset_name or f"phase_a_terrain_ranker_{args.terrain}_{args.profile}_{best.get('preset','selected')}"
        out = Path(args.out_config)
        out.parent.mkdir(parents=True, exist_ok=True)

        text = f"""version: phase_a_terrain_aware_ranker_v0
description: >
  Reference sweep config exported from Phase-A terrain-aware candidate ranker.
  Terrain and actual Gazebo world must be verified by strict world check.

terrains:
  - {args.terrain}

presets:
  - name: {preset_name}
    vx: {vx:.6f}
    yaw_rate: {yaw:.6f}
    body_height: {h:.6f}
    swing_clearance: {c:.6f}
    enable: 1.0
"""
        out.write_text(text, encoding="utf-8")
        print()
        print(f"[TRACER] wrote config: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
