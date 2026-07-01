#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/phase_a_meta_policy_bc_v0/phase_a_meta_policy_bc_v0.pt")
    ap.add_argument("--profile", default="balanced", choices=sorted(PROFILE_BETA))
    ap.add_argument("--cmd-vx", type=float, default=0.08)
    ap.add_argument("--cmd-yaw-rate", type=float, default=0.0)
    ap.add_argument("--terrain-flat", type=float, default=1.0)
    ap.add_argument("--stable-frac", type=float, default=0.67)
    ap.add_argument("--risk-proxy", type=float, default=0.33)
    ap.add_argument("--out-config", default="")
    ap.add_argument("--preset-name", default="")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    model = PhaseAPolicy(
        in_dim=int(ckpt.get("input_dim", 8)),
        out_dim=int(ckpt.get("output_dim", 3)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    beta = PROFILE_BETA[args.profile]

    x = torch.tensor([[
        beta[0],
        beta[1],
        beta[2],
        args.cmd_vx,
        args.cmd_yaw_rate,
        args.terrain_flat,
        args.stable_frac,
        args.risk_proxy,
    ]], dtype=torch.float32)

    with torch.no_grad():
        y = model(x)[0].tolist()

    vx = clamp(y[0], -0.02, 0.12)
    body_height = clamp(y[1], 0.28, 0.35)
    swing_clearance = clamp(y[2], 0.02, 0.08)

    result = {
        "profile": args.profile,
        "beta": beta,
        "input": x[0].tolist(),
        "raw_prediction": y,
        "clamped_action_phase_a": {
            "vx": vx,
            "yaw_rate": args.cmd_yaw_rate,
            "body_height": body_height,
            "swing_clearance": swing_clearance,
            "enable": 1.0,
        },
        "note": (
            "Phase-A BC smoke policy query. Current model has low action diversity "
            "and should be treated as runtime plumbing validation, not final planner."
        ),
    }

    print(json.dumps(result, indent=2))

    if args.out_config:
        preset = args.preset_name or f"phase_a_bc_{args.profile}_vx{vx:.3f}_h{body_height:.3f}_c{swing_clearance:.3f}"
        out = Path(args.out_config)
        out.parent.mkdir(parents=True, exist_ok=True)

        text = f"""version: phase_a_bc_smoke_v0
description: >
  Reference sweep config exported from Phase-A BC smoke policy.
  Layout: [counter, vx, yaw_rate, body_height, swing_clearance, enable]

terrains:
  - flat_normal

presets:
  - name: {preset}
    vx: {vx:.6f}
    yaw_rate: {args.cmd_yaw_rate:.6f}
    body_height: {body_height:.6f}
    swing_clearance: {swing_clearance:.6f}
    enable: 1.0
"""
        out.write_text(text, encoding="utf-8")
        print()
        print(f"[TRACER] wrote config: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
