#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def terrain_onehot(terrain: str, vocab: list[str]) -> list[float]:
    return [1.0 if terrain == t else 0.0 for t in vocab]


def context_features(row: dict[str, Any], terrain_vocab: list[str]) -> list[float]:
    terrain = str(row.get("terrain", "unknown"))
    return [
        *terrain_onehot(terrain, terrain_vocab),

        to_float(row.get("a_hl_vx")),
        to_float(row.get("a_hl_yaw_rate")),
        to_float(row.get("a_hl_body_height")),
        to_float(row.get("a_hl_swing_clearance")),

        to_float(row.get("stable_frac")),
        to_float(row.get("zmin_mean")),
        to_float(row.get("below22_mean")),
        to_float(row.get("distance_mean")),
        to_float(row.get("rollmax_mean")),
        to_float(row.get("pitchmax_mean")),

        to_float(row.get("ram_vertical_sink_target")),
        to_float(row.get("ram_progress_mismatch_target")),
        to_float(row.get("ram_orientation_risk_target")),
        to_float(row.get("ram_low_height_risk_target")),
        to_float(row.get("ram_recovery_needed_target")),
        to_float(row.get("ram_uncertainty_proxy_target")),
    ]


def make_x(row: dict[str, Any], terrain_vocab: list[str], side: str) -> list[float]:
    assert side in {"preferred", "rejected"}
    if side == "preferred":
        beta = [
            to_float(row.get("preferred_beta_motion")),
            to_float(row.get("preferred_beta_stability")),
            to_float(row.get("preferred_beta_energy")),
        ]
    else:
        beta = [
            to_float(row.get("rejected_beta_motion")),
            to_float(row.get("rejected_beta_stability")),
            to_float(row.get("rejected_beta_energy")),
        ]

    return [
        *context_features(row, terrain_vocab),
        *beta,
    ]


class ObjectiveUtility(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pairs-csv",
        default="data/phase_a_objective_preferences_v0/phase_a_objective_preference_pairs_v0.csv",
    )
    ap.add_argument("--out-dir", default="artifacts/phase_a_objective_selector_pref_v0")
    ap.add_argument("--terrain-vocab", default=",".join(DEFAULT_TERRAINS))
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    terrain_vocab = [x.strip() for x in args.terrain_vocab.split(",") if x.strip()]
    rows = read_csv(Path(args.pairs_csv))
    if not rows:
        raise RuntimeError(f"No pairs found: {args.pairs_csv}")

    xp = torch.tensor([make_x(r, terrain_vocab, "preferred") for r in rows], dtype=torch.float32)
    xr = torch.tensor([make_x(r, terrain_vocab, "rejected") for r in rows], dtype=torch.float32)
    weights = torch.tensor([max(0.1, to_float(r.get("pair_weight"), 1.0)) for r in rows], dtype=torch.float32)

    model = ObjectiveUtility(in_dim=xp.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        up = model(xp)
        ur = model(xr)

        # Bradley-Terry / logistic preference loss.
        loss_each = F.softplus(-(up - ur))
        loss = (loss_each * weights).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep == 1 or ep % 300 == 0 or ep == args.epochs:
            with torch.no_grad():
                acc = (up > ur).float().mean().item()
                margin = (up - ur).mean().item()
            print(f"epoch={ep:04d} loss={loss.item():.8f} pref_acc={acc:.4f} mean_margin={margin:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_schema = [
        *[f"terrain_onehot:{t}" for t in terrain_vocab],
        "a_hl_vx",
        "a_hl_yaw_rate",
        "a_hl_body_height",
        "a_hl_swing_clearance",
        "stable_frac",
        "zmin_mean",
        "below22_mean",
        "distance_mean",
        "rollmax_mean",
        "pitchmax_mean",
        "ram_vertical_sink_target",
        "ram_progress_mismatch_target",
        "ram_orientation_risk_target",
        "ram_low_height_risk_target",
        "ram_recovery_needed_target",
        "ram_uncertainty_proxy_target",
        "candidate_beta_motion",
        "candidate_beta_stability",
        "candidate_beta_energy",
    ]

    model_path = out_dir / "phase_a_objective_selector_pref_v0.pt"
    meta_path = out_dir / "phase_a_objective_selector_pref_v0_meta.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(xp.shape[1]),
            "terrain_vocab": terrain_vocab,
            "input_schema": input_schema,
            "note": (
                "Phase-A bootstrapped preference-based Objective Selector utility model. "
                "Scores candidate beta under terrain/action/RAM context."
            ),
        },
        model_path,
    )

    with torch.no_grad():
        up = model(xp)
        ur = model(xr)
        acc = (up > ur).float().mean().item()

    summary = {
        "pairs_csv": args.pairs_csv,
        "num_pairs": len(rows),
        "terrain_vocab": terrain_vocab,
        "input_dim": int(xp.shape[1]),
        "preference_accuracy": acc,
        "model_path": str(model_path),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"[TRACER] wrote model: {model_path}")
    print(f"[TRACER] wrote meta:  {meta_path}")
    print(f"[TRACER] preference_accuracy={acc:.4f}")

    print()
    print("===== sample preference margins =====")
    with torch.no_grad():
        margins = (up - ur).detach()
        order = torch.argsort(margins).tolist()

    for i in order[:10]:
        r = rows[i]
        print(
            f"LOW margin={margins[i].item():.4f} "
            f"terrain={r.get('terrain','')} "
            f"pref={r.get('preferred_profile','')} "
            f"rej={r.get('rejected_profile','')} "
            f"diff={to_float(r.get('score_diff')):.3f}"
        )

    for i in reversed(order[-10:]):
        r = rows[i]
        print(
            f"HIGH margin={margins[i].item():.4f} "
            f"terrain={r.get('terrain','')} "
            f"pref={r.get('preferred_profile','')} "
            f"rej={r.get('rejected_profile','')} "
            f"diff={to_float(r.get('score_diff')):.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
