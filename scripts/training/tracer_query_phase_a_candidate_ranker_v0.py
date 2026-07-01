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


class CandidateRanker(nn.Module):
    def __init__(self, in_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def unique_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()

    for r in rows:
        key = (
            r.get("terrain", "flat_normal"),
            r.get("preset", ""),
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


def make_x(beta: list[float], row: dict[str, Any]) -> list[float]:
    terrain = str(row.get("terrain", "flat_normal"))
    terrain_flat = 1.0 if terrain in {"flat_normal", "earth"} else 0.0

    return [
        beta[0],
        beta[1],
        beta[2],
        to_float(row.get("ref_vx")),
        to_float(row.get("ref_yaw_rate")),
        to_float(row.get("ref_body_height")),
        to_float(row.get("ref_swing_clearance")),
        terrain_flat,
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/phase_a_candidate_ranker_earth_v0/phase_a_candidate_ranker_v0.pt")
    ap.add_argument("--candidate-csv", default="data/phase_a_reference_banks_earth_v0/phase_a_reference_candidates_merged_v0.csv")
    ap.add_argument("--profile", default="balanced", choices=sorted(PROFILE_BETA))
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--out-config", default="")
    ap.add_argument("--preset-name", default="")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    model = CandidateRanker(in_dim=int(ckpt.get("input_dim", 8)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = unique_candidates(read_csv(Path(args.candidate_csv)))
    beta = PROFILE_BETA[args.profile]

    xs = torch.tensor([make_x(beta, r) for r in rows], dtype=torch.float32)

    with torch.no_grad():
        scores = model(xs).detach().tolist()

    ranked = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)

    print(f"[TRACER] profile={args.profile} beta={beta}")
    print(f"[TRACER] candidates={len(rows)}")
    print()
    print("===== ranked candidates =====")
    for i, (r, s) in enumerate(ranked[: args.top_k], start=1):
        print(
            f"{i:02d}. score={s:.4f} "
            f"preset={r.get('preset',''):32s} "
            f"vx={to_float(r.get('ref_vx')):.3f} "
            f"yaw={to_float(r.get('ref_yaw_rate')):.3f} "
            f"h={to_float(r.get('ref_body_height')):.3f} "
            f"c={to_float(r.get('ref_swing_clearance')):.3f}"
        )

    best, best_score = ranked[0]
    vx = to_float(best.get("ref_vx"))
    yaw = to_float(best.get("ref_yaw_rate"))
    h = to_float(best.get("ref_body_height"))
    c = to_float(best.get("ref_swing_clearance"))

    result = {
        "profile": args.profile,
        "beta": beta,
        "selected": {
            "preset": best.get("preset", ""),
            "score": best_score,
            "vx": vx,
            "yaw_rate": yaw,
            "body_height": h,
            "swing_clearance": c,
            "enable": 1.0,
        },
        "note": "Selected by Phase-A candidate ranker over candidate reference bank.",
    }

    print()
    print(json.dumps(result, indent=2))

    if args.out_config:
        preset_name = args.preset_name or f"phase_a_ranker_{args.profile}_{best.get('preset','selected')}"
        out = Path(args.out_config)
        out.parent.mkdir(parents=True, exist_ok=True)

        text = f"""version: phase_a_candidate_ranker_v0
description: >
  Reference sweep config exported from Phase-A candidate ranker.
  Layout: [counter, vx, yaw_rate, body_height, swing_clearance, enable]

terrains:
  - flat_normal

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
