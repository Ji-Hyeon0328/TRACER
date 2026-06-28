#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.training.tracer_train_preference_objective_irl_v0 import (
    FEATURE_NAMES,
    candidate_features,
)


def score(model: dict, x: np.ndarray) -> float:
    w = np.asarray(model["weights"], dtype=np.float64)
    b = float(model["bias"])
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    xs = (x - mean) / std
    return float(xs @ w + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="artifacts/tracer_preference_objective_irl_v0/model.json",
    )
    ap.add_argument(
        "--aggregate-csv",
        default="reports/tracer_world_style_sweep_3x3_v0_aggregate_by_style.csv",
    )
    ap.add_argument(
        "--out-csv",
        default="reports/tracer_preference_objective_irl_v0_style_scores.csv",
    )
    args = ap.parse_args()

    model = json.loads(Path(args.model).read_text())

    # Load aggregate rows by terrain/style.
    agg = {}
    with open(args.aggregate_csv, newline="") as fp:
        for r in csv.DictReader(fp):
            terrain = r["terrain"]
            style = r["style"]
            agg[(terrain, style)] = r

    objectives = {
        "motion_objective": {"beta_motion": 0.70, "beta_stability": 0.20, "beta_energy": 0.10},
        "stability_objective": {"beta_motion": 0.15, "beta_stability": 0.75, "beta_energy": 0.10},
        "deploy_objective": {"beta_motion": 0.34, "beta_stability": 0.56, "beta_energy": 0.10},
    }

    styles = ["fast", "cautious", "high_clearance"]
    terrains = sorted({k[0] for k in agg.keys()})

    rows = []
    for terrain in terrains:
        print(f"[{terrain}]")
        for objective_name, beta in objectives.items():
            scored = []
            for style in styles:
                ar = agg.get((terrain, style))
                if ar is None:
                    continue

                # Build a pseudo pair row where candidate_features can read
                # the candidate from the winner_* prefix.
                row = {
                    "terrain": terrain,
                    "objective_name": objective_name,
                    **beta,
                    "winner_success_rate": ar.get("success_rate_mean", ar.get("success_rate", 0.0)),
                    "winner_distance_mean": ar.get("distance_mean", 0.0),
                    "winner_progress_mean": ar.get("progress_mean", 0.0),
                    "winner_deploy_mean": ar.get("deploy_mean", 0.0),
                    "winner_fallen_p90_max": ar.get("fallen_p90_max", 0.0),
                    "winner_fresh1_min": ar.get("fresh1_min", 0.0),
                }

                x = candidate_features(row, style, "winner")
                s = score(model, x)
                scored.append((style, s))

            scored.sort(key=lambda kv: kv[1], reverse=True)
            print(
                f"  {objective_name:20s}: "
                + " > ".join([f"{style}({s:+.3f})" for style, s in scored])
            )

            for rank, (style, s) in enumerate(scored, start=1):
                rows.append(
                    {
                        "terrain": terrain,
                        "objective_name": objective_name,
                        "rank": rank,
                        "style": style,
                        "score": s,
                    }
                )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["terrain", "objective_name", "rank", "style", "score"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print()
    print(f"[TRACER] wrote {out}")


if __name__ == "__main__":
    main()
