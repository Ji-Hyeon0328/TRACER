#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn


class ThetaPolicyMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def load_rows(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/meta_gait_theta_dataset/theta_policy_warmstart_v0.jsonl")
    ap.add_argument("--model", default="artifacts/theta_policy_mlp_v0/theta_policy_mlp_v0.pt")
    args = ap.parse_args()

    rows = load_rows(Path(args.data))
    ckpt = torch.load(args.model, map_location="cpu")

    model = ThetaPolicyMLP(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    xs = torch.tensor([r["x"] for r in rows], dtype=torch.float32)
    ys = torch.tensor([r["theta"] for r in rows], dtype=torch.float32)

    with torch.no_grad():
        pred = model(xs)
        err = (pred - ys) ** 2
        mse = err.mean(dim=1)

    groups = defaultdict(list)
    for i, r in enumerate(rows):
        key = (
            r.get("terrain", "unknown"),
            str(r.get("gate_level_code", "unknown")),
            str(r.get("gate_action_code", "unknown")),
        )
        groups[key].append(i)

    print(f"[TRACER] n={len(rows)}")
    print(f"[TRACER] overall_mse={float(mse.mean()):.8f}")
    print()

    for key, idxs in sorted(groups.items()):
        g_mse = float(mse[idxs].mean())
        terrain, gate_level, gate_action = key
        print(
            f"========== terrain={terrain} gate={gate_level}/{gate_action} "
            f"n={len(idxs)} mse={g_mse:.8f} =========="
        )

        j = idxs[0]
        print("target theta_norm:")
        print([round(float(v), 4) for v in ys[j]])
        print("pred theta_norm:")
        print([round(float(v), 4) for v in pred[j]])
        print("target theta_physical:")
        print([round(float(v), 4) for v in rows[j]["theta_physical"]])
        print()

if __name__ == "__main__":
    main()
