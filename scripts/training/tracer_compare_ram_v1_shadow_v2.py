#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn


class RamShadowMLPv2(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ram_shadow_dataset_v2/ram_shadow_windows_v2.jsonl")
    ap.add_argument("--model", default="artifacts/ram_shadow_v2/ram_shadow_v2.pt")
    ap.add_argument("--out-csv", default="reports/ram_v1_vs_shadow_v2.csv")
    ap.add_argument("--out-json", default="reports/ram_v1_vs_shadow_v2_summary.json")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.data))
    if not rows:
        raise RuntimeError(f"no rows found: {args.data}")

    ckpt = torch.load(args.model, map_location="cpu")
    input_dim = int(ckpt["input_dim"])
    output_dim = int(ckpt["output_dim"])
    hidden = int(ckpt.get("hidden_dim", 256))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RamShadowMLPv2(input_dim, output_dim, hidden).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32, device=device).view(1, -1)
    std = torch.tensor(ckpt["x_std"], dtype=torch.float32, device=device).view(1, -1).clamp_min(1e-6)

    label_names = ckpt["label_names"]
    feature_names = ckpt["feature_names"]

    per_step = len(feature_names)
    idx_v1_intervention = feature_names.index("learned_ram_intervention_score")
    idx_v1_override = feature_names.index("learned_ram_future_override_mean")

    out_rows = []

    for r in rows:
        x = torch.tensor(r["x"], dtype=torch.float32, device=device).view(1, -1)
        xn = (x - mean) / std
        pred = torch.sigmoid(model(xn))[0].detach().cpu().tolist()

        # Last step in the input window approximates current RAM v1 signal.
        last = r["x"][-per_step:]
        v1_intervention = f(last[idx_v1_intervention])
        v1_override = f(last[idx_v1_override])

        y = r["y"]

        out = {
            "terrain": r.get("terrain", "unknown"),
            "source_csv": r.get("source_csv", ""),
            "end_index": r.get("end_index", -1),
            "ram_v1_current_intervention": v1_intervention,
            "ram_v1_current_override": v1_override,
            "ram_v2_future_intervention": pred[label_names.index("future_intervention")],
            "ram_v2_future_override_mean": pred[label_names.index("future_override_mean")],
            "label_future_intervention": y[label_names.index("future_intervention")],
            "label_future_override_mean": y[label_names.index("future_override_mean")],
            "label_future_conservative": y[label_names.index("future_conservative")],
            "label_future_low_speed": y[label_names.index("future_low_speed")],
            "label_future_gms_disagree": y[label_names.index("future_gms_disagree")],
            "label_episode_success": y[label_names.index("episode_success")],
        }
        out_rows.append(out)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if out_rows:
        with out_csv.open("w", newline="") as fobj:
            w = csv.DictWriter(fobj, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)

    groups = defaultdict(list)
    for r in out_rows:
        groups[r["terrain"]].append(r)

    summary = {}
    for terrain, rs in groups.items():
        def mean_key(k):
            return sum(f(r[k]) for r in rs) / max(1, len(rs))

        summary[terrain] = {
            "n": len(rs),
            "ram_v1_current_intervention_mean": mean_key("ram_v1_current_intervention"),
            "ram_v1_current_override_mean": mean_key("ram_v1_current_override"),
            "ram_v2_future_intervention_mean": mean_key("ram_v2_future_intervention"),
            "ram_v2_future_override_mean": mean_key("ram_v2_future_override_mean"),
            "label_future_intervention_mean": mean_key("label_future_intervention"),
            "label_future_override_mean": mean_key("label_future_override_mean"),
            "label_future_conservative_mean": mean_key("label_future_conservative"),
            "label_future_low_speed_mean": mean_key("label_future_low_speed"),
            "label_future_gms_disagree_mean": mean_key("label_future_gms_disagree"),
            "label_episode_success_mean": mean_key("label_episode_success"),
        }

    out_json.write_text(json.dumps(summary, indent=2))

    print("[TRACER] wrote RAM v1 vs RAM shadow v2 comparison")
    print(json.dumps({
        "out_csv": str(out_csv),
        "out_json": str(out_json),
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
