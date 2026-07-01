#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def parse_beta(s: Any) -> list[float]:
    if isinstance(s, list):
        return [float(x) for x in s]
    try:
        return [float(x) for x in json.loads(str(s))]
    except Exception:
        return [0.34, 0.33, 0.33]


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


def row_to_x(row: dict[str, Any]) -> list[float]:
    beta = parse_beta(row.get("beta", ""))

    # Phase-A ranker feature:
    # beta(3) + candidate action(4) + terrain_flat_stub(1)
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


def build_xy(rows: list[dict[str, Any]]):
    xs = []
    ys = []
    kept = []

    for r in rows:
        if "R_gated_mean" not in r:
            continue
        y = max(0.0, min(1.0, to_float(r.get("R_gated_mean"))))
        x = row_to_x(r)
        xs.append(x)
        ys.append(y)
        kept.append(r)

    if not xs:
        raise RuntimeError("No usable rows found. Need columns beta/ref_vx/ref_body_height/ref_swing_clearance/R_gated_mean.")

    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.float32),
        kept,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate-csv",
        default="data/phase_a_reference_banks_earth_v0/phase_a_reference_candidates_merged_v0.csv",
    )
    ap.add_argument("--out-dir", default="artifacts/phase_a_candidate_ranker_earth_v0")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    rows = read_csv(Path(args.candidate_csv))
    x, y, kept = build_xy(rows)

    model = CandidateRanker(in_dim=x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, args.epochs + 1):
        pred = model(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep == 1 or ep % 200 == 0 or ep == args.epochs:
            mae = (pred.detach() - y).abs().mean().item()
            print(f"epoch={ep:04d} loss={loss.item():.8f} mae_score={mae:.5f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "phase_a_candidate_ranker_v0.pt"
    meta_path = out_dir / "phase_a_candidate_ranker_v0_meta.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": x.shape[1],
            "input_schema": [
                "beta_motion",
                "beta_stability",
                "beta_energy",
                "candidate_vx",
                "candidate_yaw_rate",
                "candidate_body_height",
                "candidate_swing_clearance",
                "terrain_flat_stub",
            ],
            "target_schema": ["R_gated_mean"],
            "note": (
                "Phase-A candidate ranker. Scores candidate reference actions "
                "conditioned on objective beta and terrain stub. Use for argmax "
                "selection over a candidate bank, not direct continuous action regression."
            ),
        },
        model_path,
    )

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "candidate_csv": args.candidate_csv,
                "num_samples": len(kept),
                "x_mean": x.mean(dim=0).tolist(),
                "y_mean": float(y.mean().item()),
                "y_min": float(y.min().item()),
                "y_max": float(y.max().item()),
                "model_path": str(model_path),
            },
            f,
            indent=2,
        )

    print()
    print(f"[TRACER] wrote model: {model_path}")
    print(f"[TRACER] wrote meta:  {meta_path}")
    print()

    with torch.no_grad():
        pred = model(x).detach()

    order = torch.argsort(pred, descending=True).tolist()

    print("===== top predicted candidates =====")
    for idx in order[:20]:
        r = kept[idx]
        print(
            f"profile={r.get('profile',''):18s} "
            f"preset={r.get('preset',''):32s} "
            f"true={y[idx].item():.4f} pred={pred[idx].item():.4f} "
            f"vx={to_float(r.get('ref_vx')):.3f} "
            f"h={to_float(r.get('ref_body_height')):.3f} "
            f"c={to_float(r.get('ref_swing_clearance')):.3f}"
        )

    print()
    print("===== largest absolute errors =====")
    err_order = torch.argsort((pred - y).abs(), descending=True).tolist()
    for idx in err_order[:20]:
        r = kept[idx]
        print(
            f"profile={r.get('profile',''):18s} "
            f"preset={r.get('preset',''):32s} "
            f"true={y[idx].item():.4f} pred={pred[idx].item():.4f} "
            f"err={abs(pred[idx].item()-y[idx].item()):.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
