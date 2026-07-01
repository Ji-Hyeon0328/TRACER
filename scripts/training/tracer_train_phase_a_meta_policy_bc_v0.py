#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


class PhaseAPolicy(nn.Module):
    def __init__(self, in_dim: int = 8, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_xy(rows):
    xs = []
    ys = []

    for r in rows:
        beta = r.get("beta", [0.34, 0.33, 0.33])
        if isinstance(beta, str):
            beta = json.loads(beta)

        a = r["action_phase_a"]
        m = r.get("metrics", {})

        # Phase-A minimal input:
        # beta(3) + task command stub(2) + terrain flat stub(1) + reliability stats(2)
        # This is intentionally small and will be expanded later with RAM/context/goal.
        cmd_vx = float(a.get("vx", 0.0))
        cmd_yaw = float(a.get("yaw_rate", 0.0))
        terrain_flat = 1.0

        stable_frac = float(m.get("stable_frac", 0.0))
        risk_proxy = 1.0 - stable_frac

        x = [
            float(beta[0]),
            float(beta[1]),
            float(beta[2]),
            cmd_vx,
            cmd_yaw,
            terrain_flat,
            stable_frac,
            risk_proxy,
        ]

        y = [
            float(a["vx"]),
            float(a["body_height"]),
            float(a["swing_clearance"]),
        ]

        xs.append(x)
        ys.append(y)

    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--positive-bank",
        default="data/phase_a_reference_banks_v0/phase_a_positive_teacher_bank_v0.jsonl",
    )
    ap.add_argument("--out-dir", default="artifacts/phase_a_meta_policy_bc_v0")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.positive_bank))
    if not rows:
        raise RuntimeError(f"No positive teacher rows found: {args.positive_bank}")

    x, y = build_xy(rows)

    model = PhaseAPolicy(in_dim=x.shape[1], out_dim=y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, args.epochs + 1):
        pred = model(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep == 1 or ep % 100 == 0 or ep == args.epochs:
            mae = (pred.detach() - y).abs().mean(dim=0)
            print(
                f"epoch={ep:04d} loss={loss.item():.8f} "
                f"mae_vx={mae[0].item():.5f} "
                f"mae_h={mae[1].item():.5f} "
                f"mae_clear={mae[2].item():.5f}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "phase_a_meta_policy_bc_v0.pt"
    meta_path = out_dir / "phase_a_meta_policy_bc_v0_meta.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": x.shape[1],
            "output_dim": y.shape[1],
            "input_schema": [
                "beta_motion",
                "beta_stability",
                "beta_energy",
                "cmd_vx",
                "cmd_yaw_rate",
                "terrain_flat",
                "stable_frac",
                "risk_proxy",
            ],
            "output_schema": [
                "vx",
                "body_height",
                "swing_clearance",
            ],
            "note": (
                "Phase-A BC smoke model. Current positive bank has low action diversity; "
                "use only as warm-start/runtime plumbing validation."
            ),
        },
        model_path,
    )

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "positive_bank": args.positive_bank,
                "num_samples": len(rows),
                "x_mean": x.mean(dim=0).tolist(),
                "y_mean": y.mean(dim=0).tolist(),
                "model_path": str(model_path),
                "note": "BC smoke model for Phase-A meta-action [vx, body_height, swing_clearance].",
            },
            f,
            indent=2,
        )

    print()
    print(f"[TRACER] wrote model: {model_path}")
    print(f"[TRACER] wrote meta:  {meta_path}")

    with torch.no_grad():
        pred = model(x)
        print()
        print("[TRACER] sample predictions:")
        for i in range(min(10, len(rows))):
            print(
                f"{rows[i]['profile']:18s} "
                f"target={y[i].tolist()} pred={pred[i].tolist()}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
