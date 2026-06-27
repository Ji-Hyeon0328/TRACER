#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List


def f(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def terrain_onehot(name: str) -> List[float]:
    return [
        1.0 if name == "flat_normal" else 0.0,
        1.0 if name == "rough_mid" else 0.0,
        1.0 if name == "slope_5deg" else 0.0,
    ]


def terrain_prior_beta(terrain: str) -> List[float]:
    if terrain == "flat_normal":
        return [0.55, 0.25, 0.20]
    if terrain == "rough_mid":
        return [0.30, 0.50, 0.20]
    if terrain == "slope_5deg":
        return [0.25, 0.55, 0.20]
    return [0.25, 0.55, 0.20]


FEATURE_NAMES = [
    "terrain_flat_normal",
    "terrain_rough_mid",
    "terrain_slope_5deg",
    "duration_norm",
    "vx_over_target",
    "enable_mean",
    "body_height_delta",
    "clearance_delta",
    "gate_level_norm",
    "gate_action_norm",
    "gate_override_mean",
    "ram_fallen_weak",
    "ram_recovery",
    "success",
    "preference_score_norm",
]

TARGET_NAMES = [
    "beta_motion",
    "beta_stability",
    "beta_energy",
]


def normalize_beta(v: List[float]) -> List[float]:
    v = [max(1e-4, float(x)) for x in v]
    s = sum(v)
    return [x / s for x in v]


def build_x(row: dict) -> List[float]:
    terrain = str(row.get("terrain", "unknown"))
    target_vx = max(1e-6, f(row.get("target_vx", 0.05)))
    score = f(row.get("pred_preference_score_v2", row.get("total_score", 0.0)))

    # Preference model scores are unbounded. This squash only makes them a stable input.
    score_norm = 1.0 / (1.0 + math.exp(-score / 4.0))

    return terrain_onehot(terrain) + [
        clip(f(row.get("duration_sec", 0.0)) / 30.0),
        clip(f(row.get("mpc_vx_mean", 0.0)) / target_vx, 0.0, 2.0) / 2.0,
        clip(f(row.get("mpc_enable_mean", 0.0))),
        f(row.get("mpc_body_height_mean", 0.295)) - 0.295,
        f(row.get("mpc_clearance_mean", 0.030)) - 0.030,
        clip(f(row.get("gate_level_mean", 0.0)) / 2.0),
        clip(f(row.get("gate_action_mean", 0.0)) / 2.0),
        clip(f(row.get("gate_override_mean", 0.0))),
        clip(f(row.get("ram_run_fallen_mean", 0.0))),
        clip(f(row.get("ram_recovery_needed_mean", 0.0))),
        1.0 if str(row.get("success_proxy", "False")).lower() == "true" else 0.0,
        score_norm,
    ]


def build_beta_target(row: dict) -> List[float]:
    terrain = str(row.get("terrain", "unknown"))
    beta = terrain_prior_beta(terrain)

    gate_level = clip(f(row.get("gate_level_mean", 0.0)) / 2.0)
    gate_override = clip(f(row.get("gate_override_mean", 0.0)))
    recovery = clip(f(row.get("ram_recovery_needed_mean", 0.0)))

    target_vx = max(1e-6, f(row.get("target_vx", 0.05)))
    vx_ratio = clip(f(row.get("mpc_vx_mean", 0.0)) / target_vx, 0.0, 1.5)

    h_effort = abs(f(row.get("mpc_body_height_mean", 0.295)) - 0.295) / 0.08
    c_effort = abs(f(row.get("mpc_clearance_mean", 0.030)) - 0.030) / 0.12
    effort = clip((h_effort + c_effort) / 2.0)

    success = 1.0 if str(row.get("success_proxy", "False")).lower() == "true" else 0.0

    # Rule:
    # - More gate intervention / recovery -> emphasize stability.
    # - Good speed with low intervention -> allow more motion.
    # - High command effort / frequent override -> reduce energy weight.
    # This is still a warm-start target, not a final human preference label.
    stability_boost = 0.25 * gate_level + 0.25 * gate_override + 0.10 * recovery
    motion_boost = 0.15 * success * max(0.0, vx_ratio - 0.8) * (1.0 - gate_override)
    energy_penalty = 0.15 * effort + 0.10 * gate_override

    beta[1] += stability_boost
    beta[0] += motion_boost
    beta[2] -= energy_penalty

    # If rollout still scores highly despite gate intervention, don't over-penalize motion.
    total_score = f(row.get("total_score", 0.0))
    if total_score > 0.75 and success > 0.5:
        beta[0] += 0.05
        beta[1] -= 0.03

    return normalize_beta(beta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-csv", default="artifacts/objective_preference_v2/scored_episodes_v2.csv")
    ap.add_argument("--out", default="data/objective_selector_dataset_v2/objective_selector_dataset_v2.jsonl")
    args = ap.parse_args()

    scored_csv = Path(args.scored_csv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    with scored_csv.open() as fobj:
        for row in csv.DictReader(fobj):
            x = build_x(row)
            y = build_beta_target(row)

            samples.append({
                "episode_id": row.get("episode_id", ""),
                "terrain": row.get("terrain", "unknown"),
                "policy_id": row.get("policy_id", ""),
                "x": x,
                "y": y,
                "feature_names": FEATURE_NAMES,
                "target_names": TARGET_NAMES,
                "source": "objective_selector_dataset_v2_from_preference_scored_rollouts",
                "notes": "Warm-start beta target generated from terrain prior, preference score, gate intervention, and command effort.",
            })

    with out.open("w") as fobj:
        for s in samples:
            fobj.write(json.dumps(s) + "\n")

    by_terrain = {}
    for s in samples:
        by_terrain.setdefault(s["terrain"], 0)
        by_terrain[s["terrain"]] += 1

    manifest = {
        "version": "objective_selector_dataset_v2",
        "n_samples": len(samples),
        "input_dim": len(FEATURE_NAMES),
        "output_dim": len(TARGET_NAMES),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "by_terrain": by_terrain,
        "scored_csv": str(scored_csv),
        "out": str(out),
    }
    manifest_path = out.parent / "manifest_v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("[TRACER] wrote objective selector dataset v2")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
