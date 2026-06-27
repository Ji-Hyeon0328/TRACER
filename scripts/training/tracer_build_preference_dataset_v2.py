#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List


def f(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def terrain_target_vx(terrain: str) -> float:
    if terrain == "flat_normal":
        return 0.28
    if terrain == "rough_mid":
        return 0.055
    if terrain == "slope_5deg":
        return 0.045
    return 0.05


def terrain_beta_prior(terrain: str) -> List[float]:
    if terrain == "flat_normal":
        return [0.55, 0.25, 0.20]
    if terrain == "rough_mid":
        return [0.30, 0.50, 0.20]
    if terrain == "slope_5deg":
        return [0.25, 0.55, 0.20]
    return [0.25, 0.55, 0.20]


def compute_scores(s: dict) -> dict:
    terrain = str(s.get("terrain", "unknown"))
    target_vx = terrain_target_vx(terrain)

    success = 1.0 if bool(s.get("success_proxy", False)) else 0.0

    vx_mean = f(s.get("mpc_vx_mean", 0.0))
    enable_mean = f(s.get("mpc_enable_mean", 0.0))
    gate_level = f(s.get("gate_level_mean", 0.0))
    gate_override = f(s.get("gate_override_mean", 0.0))
    fallen = f(s.get("ram_run_fallen_mean", 0.0))
    recovery = f(s.get("ram_recovery_needed_mean", 0.0))

    # Existing RAM fallen is known to be poorly calibrated, so we use it as a weak penalty.
    # Gate override is more meaningful for current runtime behavior.
    motion_score = success * enable_mean * clip(vx_mean / max(1e-6, target_vx), 0.0, 1.5) / 1.5
    stability_score = success * clip(
        1.0
        - 0.18 * clip(gate_level / 2.0)
        - 0.35 * clip(gate_override)
        - 0.10 * clip(fallen)
        - 0.20 * clip(recovery),
        0.0,
        1.0,
    )

    # Until torque/energy is available, this is a conservative proxy:
    # less intervention and not excessively high body/clearance command is preferred.
    h = f(s.get("mpc_body_height_mean", 0.0))
    clr = f(s.get("mpc_clearance_mean", 0.0))
    command_effort = clip(abs(h - 0.295) / 0.08 + abs(clr - 0.03) / 0.12, 0.0, 1.5) / 1.5
    energy_score = success * clip(1.0 - 0.45 * command_effort - 0.25 * clip(gate_override), 0.0, 1.0)

    beta = terrain_beta_prior(terrain)
    total_score = (
        beta[0] * motion_score
        + beta[1] * stability_score
        + beta[2] * energy_score
    )

    return {
        "target_vx": target_vx,
        "success": success,
        "motion_score": motion_score,
        "stability_score": stability_score,
        "energy_score": energy_score,
        "total_score": total_score,
        "beta_motion": beta[0],
        "beta_stability": beta[1],
        "beta_energy": beta[2],
    }


def row_from_summary(path: Path) -> dict:
    s = load_json(path)
    scores = compute_scores(s)

    row = {
        "episode_id": s.get("episode_id", path.stem),
        "terrain": s.get("terrain", "unknown"),
        "policy_id": s.get("policy_id", ""),
        "duration_sec": f(s.get("duration_sec", 0.0)),
        "n_rows": int(f(s.get("n_rows", 0))),
        "mpc_vx_mean": f(s.get("mpc_vx_mean", 0.0)),
        "mpc_enable_mean": f(s.get("mpc_enable_mean", 0.0)),
        "mpc_body_height_mean": f(s.get("mpc_body_height_mean", 0.0)),
        "mpc_clearance_mean": f(s.get("mpc_clearance_mean", 0.0)),
        "gate_level_mean": f(s.get("gate_level_mean", 0.0)),
        "gate_action_mean": f(s.get("gate_action_mean", 0.0)),
        "gate_override_mean": f(s.get("gate_override_mean", 0.0)),
        "ram_run_fallen_mean": f(s.get("ram_run_fallen_mean", 0.0)),
        "ram_recovery_needed_mean": f(s.get("ram_recovery_needed_mean", 0.0)),
        "distance_xy_proxy": f(s.get("distance_xy_proxy", 0.0)),
        "success_proxy": bool(s.get("success_proxy", False)),
        "summary_json": str(path),
        "step_csv": s.get("step_csv", ""),
    }
    row.update(scores)
    return row


def pairwise_preferences(rows: List[dict], margin: float) -> List[dict]:
    pairs = []
    by_terrain: Dict[str, List[dict]] = {}
    for r in rows:
        by_terrain.setdefault(str(r["terrain"]), []).append(r)

    for terrain, group in by_terrain.items():
        for a, b in combinations(group, 2):
            da = f(a["total_score"])
            db = f(b["total_score"])
            diff = da - db

            if abs(diff) < margin:
                continue

            preferred, rejected = (a, b) if diff > 0 else (b, a)

            pairs.append({
                "terrain": terrain,
                "preferred_episode_id": preferred["episode_id"],
                "rejected_episode_id": rejected["episode_id"],
                "preferred_policy_id": preferred["policy_id"],
                "rejected_policy_id": rejected["policy_id"],
                "preferred_score": preferred["total_score"],
                "rejected_score": rejected["total_score"],
                "score_margin": abs(diff),
                "preference_reason": "weighted_proxy_score_v2",
                "preferred_summary_json": preferred["summary_json"],
                "rejected_summary_json": rejected["summary_json"],
            })

    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries-glob", default="data/rollout_dataset_v0/summaries/*.json")
    ap.add_argument("--out-dir", default="data/preference_datasets_v2")
    ap.add_argument("--min-duration", type=float, default=8.0)
    ap.add_argument("--margin", type=float, default=0.03)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(Path(".").glob(args.summaries_glob))
    rows = []
    for p in paths:
        try:
            r = row_from_summary(p)
        except Exception as e:
            print(f"[WARN] skip {p}: {e!r}")
            continue
        if f(r["duration_sec"]) < args.min_duration:
            continue
        rows.append(r)

    rows = sorted(rows, key=lambda r: (r["terrain"], r["policy_id"], r["episode_id"]))

    metrics_csv = out_dir / "rollout_metrics_v2.csv"
    metrics_jsonl = out_dir / "rollout_metrics_v2.jsonl"
    pairs_jsonl = out_dir / "rollout_preference_pairs_v2.jsonl"
    manifest_json = out_dir / "manifest_v2.json"

    if rows:
        with metrics_csv.open("w", newline="") as fobj:
            w = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    with metrics_jsonl.open("w") as fobj:
        for r in rows:
            fobj.write(json.dumps(r) + "\n")

    pairs = pairwise_preferences(rows, args.margin)
    with pairs_jsonl.open("w") as fobj:
        for p in pairs:
            fobj.write(json.dumps(p) + "\n")

    by_terrain = {}
    for r in rows:
        by_terrain.setdefault(r["terrain"], 0)
        by_terrain[r["terrain"]] += 1

    manifest = {
        "version": "preference_dataset_v2",
        "summaries_glob": args.summaries_glob,
        "n_episodes": len(rows),
        "n_pairs": len(pairs),
        "min_duration": args.min_duration,
        "margin": args.margin,
        "by_terrain": by_terrain,
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "metrics_jsonl": str(metrics_jsonl),
            "pairs_jsonl": str(pairs_jsonl),
        },
        "notes": [
            "RAM fallen score is currently weakly calibrated and used only as a weak penalty.",
            "Energy score is a command-effort proxy until torque/energy logging is available.",
            "Pairwise labels are proxy preferences for Objective Selector IRL warm-start.",
        ],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2))

    print("[TRACER] wrote preference dataset v2")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
