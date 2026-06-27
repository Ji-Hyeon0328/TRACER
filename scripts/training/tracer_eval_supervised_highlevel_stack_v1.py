#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn


class ObjectiveSelectorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class RamInterventionMLP(nn.Module):
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


class GMSMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_jsonl(path: str):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def norm_x(x, mean, std):
    # Keep input, mean, std on the same device.
    xt = torch.tensor(x, dtype=torch.float32, device=mean.device)
    return (xt - mean) / std.clamp_min(1e-6)


def mean(xs):
    return sum(xs) / max(1, len(xs))


def eval_objective(args, device):
    bundle = torch.load(args.objective_model, map_location="cpu")
    model = ObjectiveSelectorMLP(bundle["input_dim"], bundle.get("hidden_dim", 64))
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device).eval()

    x_mean = torch.tensor(bundle["x_mean"], dtype=torch.float32).to(device)
    x_std = torch.tensor(bundle["x_std"], dtype=torch.float32).to(device)

    rows = load_jsonl(args.objective_data)

    by_terrain = defaultdict(lambda: {"n": 0, "mae": [], "pred": [], "target": []})
    with torch.no_grad():
        for r in rows:
            x = norm_x(r["x"], x_mean, x_std).to(device).unsqueeze(0)
            pred = model(x).cpu()[0].tolist()
            target = [float(v) for v in r["beta"]]
            mae = mean([abs(a - b) for a, b in zip(pred, target)])

            t = r["terrain"]
            by_terrain[t]["n"] += 1
            by_terrain[t]["mae"].append(mae)
            by_terrain[t]["pred"].append(pred)
            by_terrain[t]["target"].append(target)

    report = {}
    for t, d in sorted(by_terrain.items()):
        pred_mean = [mean([p[i] for p in d["pred"]]) for i in range(3)]
        target_mean = [mean([p[i] for p in d["target"]]) for i in range(3)]
        report[t] = {
            "n": d["n"],
            "mae": mean(d["mae"]),
            "pred_beta_mean": pred_mean,
            "target_beta_mean": target_mean,
        }
    return report


def eval_ram(args, device):
    bundle = torch.load(args.ram_model, map_location="cpu")
    labels = bundle["label_names"]

    model = RamInterventionMLP(
        bundle["input_dim"],
        bundle["output_dim"],
        bundle.get("hidden_dim", 256),
    )
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device).eval()

    x_mean = torch.tensor(bundle["x_mean"], dtype=torch.float32).to(device)
    x_std = torch.tensor(bundle["x_std"], dtype=torch.float32).to(device)

    rows = load_jsonl(args.ram_data)

    by_terrain = defaultdict(lambda: {"n": 0, "pred": [], "target": []})
    with torch.no_grad():
        for r in rows:
            x = norm_x(r["x"], x_mean, x_std).to(device).unsqueeze(0)
            pred = torch.sigmoid(model(x)).cpu()[0].tolist()
            target = [float(r["y"].get(k, 0.0)) for k in labels]

            t = r["terrain"]
            by_terrain[t]["n"] += 1
            by_terrain[t]["pred"].append(pred)
            by_terrain[t]["target"].append(target)

    report = {}
    for t, d in sorted(by_terrain.items()):
        pred_mean = {labels[i]: mean([p[i] for p in d["pred"]]) for i in range(len(labels))}
        target_mean = {labels[i]: mean([p[i] for p in d["target"]]) for i in range(len(labels))}
        mae = mean([
            abs(pred_mean[k] - target_mean[k])
            for k in labels
        ])
        report[t] = {
            "n": d["n"],
            "mean_label_mae": mae,
            "pred_mean": pred_mean,
            "target_mean": target_mean,
        }
    return report


def eval_gms(args, device):
    bundle = torch.load(args.gms_model, map_location="cpu")
    labels = bundle["label_names"]

    model = GMSMLP(
        bundle["input_dim"],
        len(labels),
        bundle.get("hidden_dim", 128),
    )
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device).eval()

    x_mean = torch.tensor(bundle["x_mean"], dtype=torch.float32).to(device)
    x_std = torch.tensor(bundle["x_std"], dtype=torch.float32).to(device)

    rows = load_jsonl(args.gms_data)

    by_terrain = defaultdict(lambda: {"n": 0, "correct": 0, "pred_counts": defaultdict(int), "target_counts": defaultdict(int)})
    confusion = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for r in rows:
            x = norm_x(r["x"], x_mean, x_std).to(device).unsqueeze(0)
            logits = model(x)
            pred_id = int(logits.argmax(dim=-1).cpu().item())
            target_id = int(r["label_id"])

            pred = labels[pred_id]
            target = labels[target_id]
            terrain = r["terrain"]

            by_terrain[terrain]["n"] += 1
            by_terrain[terrain]["correct"] += int(pred_id == target_id)
            by_terrain[terrain]["pred_counts"][pred] += 1
            by_terrain[terrain]["target_counts"][target] += 1
            confusion[target][pred] += 1

    report = {}
    for t, d in sorted(by_terrain.items()):
        report[t] = {
            "n": d["n"],
            "acc": d["correct"] / max(1, d["n"]),
            "pred_counts": dict(d["pred_counts"]),
            "target_counts": dict(d["target_counts"]),
        }

    return {
        "by_terrain": report,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective-model", default="artifacts/objective_selector_v1/objective_selector_v1.pt")
    ap.add_argument("--objective-data", default="data/objective_selector_dataset_v1/objective_selector_dataset_v1.jsonl")
    ap.add_argument("--ram-model", default="artifacts/ram_intervention_v1/ram_intervention_v1.pt")
    ap.add_argument("--ram-data", default="data/ram_window_dataset_v1/ram_windows_v1.jsonl")
    ap.add_argument("--gms-model", default="artifacts/gms_classifier_v1/gms_classifier_v1.pt")
    ap.add_argument("--gms-data", default="data/gms_dataset_v1/gms_dataset_v1.jsonl")
    ap.add_argument("--out", default="reports/supervised_highlevel_stack_v1.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report = {
        "objective_selector": eval_objective(args, device),
        "ram_intervention": eval_ram(args, device),
        "gms_classifier": eval_gms(args, device),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"[TRACER] wrote supervised high-level stack report: {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
