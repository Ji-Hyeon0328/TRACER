#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path


OBJECTIVE_BETA = {
    # Prefer faster task progress.
    "motion_objective": {
        "motion": 0.70,
        "stability": 0.20,
        "energy": 0.10,
    },
    # Prefer safer / more reliable deployment.
    "stability_objective": {
        "motion": 0.15,
        "stability": 0.75,
        "energy": 0.10,
    },
    # Balanced deployment objective, close to current TRACER runtime use.
    "deploy_objective": {
        "motion": 0.34,
        "stability": 0.56,
        "energy": 0.10,
    },
}


def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def objective_scores_for_terrain(rows):
    """
    Compute objective-conditioned scores within one terrain.

    The normalization is terrain-local because style candidates should be compared
    under the same world/terrain condition.
    """
    max_distance = max(1e-6, max(f(r.get("distance_mean")) for r in rows))
    max_progress = max(1e-6, max(f(r.get("progress_mean")) for r in rows))

    scored = []
    for r in rows:
        distance_norm = clamp(f(r.get("distance_mean")) / max_distance)
        progress_norm = clamp(f(r.get("progress_mean")) / max_progress)

        success_rate = clamp(f(r.get("success_rate")))
        fresh_min = clamp(f(r.get("fresh1_min")))
        fallen_max = clamp(f(r.get("fallen_p90_max")))

        # Severe fallen spikes should dominate safety-related objectives.
        fallen_penalty = fallen_max

        motion_score = (
            0.75 * distance_norm
            + 0.15 * progress_norm
            + 0.10 * fresh_min
            - 0.20 * (1.0 - success_rate)
            - 0.30 * fallen_penalty
        )

        stability_score = (
            0.45 * success_rate
            + 0.25 * fresh_min
            + 0.20 * (1.0 - fallen_penalty)
            + 0.10 * progress_norm
        )

        deploy_score = f(r.get("deploy_mean"))

        out = dict(r)
        out["distance_norm"] = distance_norm
        out["progress_norm"] = progress_norm
        out["motion_score"] = motion_score
        out["stability_score"] = stability_score
        out["deploy_score"] = deploy_score
        scored.append(out)

    return scored


def confidence_from_margin(margin: float) -> float:
    # Soft confidence for small smoke datasets.
    # 0.05 -> weak, 0.25+ -> strong.
    return clamp((margin - 0.03) / 0.22)


def make_pair(terrain, objective_name, score_key, winner, loser, margin):
    beta = OBJECTIVE_BETA[objective_name]
    return {
        "terrain": terrain,
        "objective_name": objective_name,
        "beta_motion": beta["motion"],
        "beta_stability": beta["stability"],
        "beta_energy": beta["energy"],

        "winner_style": winner["style"],
        "loser_style": loser["style"],

        "winner_score": f(winner[score_key]),
        "loser_score": f(loser[score_key]),
        "margin": margin,
        "label_confidence": confidence_from_margin(margin),

        "winner_success_rate": f(winner.get("success_rate")),
        "loser_success_rate": f(loser.get("success_rate")),

        "winner_distance_mean": f(winner.get("distance_mean")),
        "loser_distance_mean": f(loser.get("distance_mean")),

        "winner_progress_mean": f(winner.get("progress_mean")),
        "loser_progress_mean": f(loser.get("progress_mean")),

        "winner_deploy_mean": f(winner.get("deploy_mean")),
        "loser_deploy_mean": f(loser.get("deploy_mean")),

        "winner_fallen_p90_max": f(winner.get("fallen_p90_max")),
        "loser_fallen_p90_max": f(loser.get("fallen_p90_max")),

        "winner_fresh1_min": f(winner.get("fresh1_min")),
        "loser_fresh1_min": f(loser.get("fresh1_min")),

        "winner_episode_ids": winner.get("episode_ids", ""),
        "loser_episode_ids": loser.get("episode_ids", ""),
        "label_source": "world_style_sweep_3x3_v0_objective_conditioned",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--aggregate-csv",
        default="reports/tracer_world_style_sweep_3x3_v0_aggregate_by_style.csv",
    )
    ap.add_argument(
        "--out-csv",
        default="data/preference_datasets/tracer_objective_conditioned_style_pairs_3x3_v0.csv",
    )
    ap.add_argument(
        "--out-jsonl",
        default="data/preference_datasets/tracer_objective_conditioned_style_pairs_3x3_v0.jsonl",
    )
    ap.add_argument("--min-margin-motion", type=float, default=0.05)
    ap.add_argument("--min-margin-stability", type=float, default=0.04)
    ap.add_argument("--min-margin-deploy", type=float, default=0.03)
    args = ap.parse_args()

    rows = []
    with open(args.aggregate_csv, newline="") as fp:
        for r in csv.DictReader(fp):
            rows.append(r)

    by_terrain = {}
    for r in rows:
        by_terrain.setdefault(r["terrain"], []).append(r)

    all_pairs = []
    scored_rows = []

    objective_specs = [
        ("motion_objective", "motion_score", args.min_margin_motion),
        ("stability_objective", "stability_score", args.min_margin_stability),
        ("deploy_objective", "deploy_score", args.min_margin_deploy),
    ]

    for terrain, terrain_rows in sorted(by_terrain.items()):
        scored = objective_scores_for_terrain(terrain_rows)
        scored_rows.extend(scored)

        for objective_name, score_key, min_margin in objective_specs:
            for a, b in combinations(scored, 2):
                qa = f(a[score_key])
                qb = f(b[score_key])
                margin = abs(qa - qb)

                if margin < min_margin:
                    continue

                winner, loser = (a, b) if qa > qb else (b, a)
                all_pairs.append(
                    make_pair(
                        terrain=terrain,
                        objective_name=objective_name,
                        score_key=score_key,
                        winner=winner,
                        loser=loser,
                        margin=margin,
                    )
                )

    out_csv = Path(args.out_csv)
    out_jsonl = Path(args.out_jsonl)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "terrain",
        "objective_name",
        "beta_motion",
        "beta_stability",
        "beta_energy",
        "winner_style",
        "loser_style",
        "winner_score",
        "loser_score",
        "margin",
        "label_confidence",
        "winner_success_rate",
        "loser_success_rate",
        "winner_distance_mean",
        "loser_distance_mean",
        "winner_progress_mean",
        "loser_progress_mean",
        "winner_deploy_mean",
        "loser_deploy_mean",
        "winner_fallen_p90_max",
        "loser_fallen_p90_max",
        "winner_fresh1_min",
        "loser_fresh1_min",
        "winner_episode_ids",
        "loser_episode_ids",
        "label_source",
    ]

    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for p in all_pairs:
            w.writerow({k: p.get(k, "") for k in fields})

    with out_jsonl.open("w") as fp:
        for p in all_pairs:
            fp.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[TRACER] aggregate_rows={len(rows)} objective_pairs={len(all_pairs)}")
    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_jsonl}")
    print()

    for terrain in sorted(by_terrain.keys()):
        print(f"[{terrain}]")
        scored = [r for r in scored_rows if r["terrain"] == terrain]
        for objective_name, score_key, _ in objective_specs:
            ordered = sorted(scored, key=lambda x: f(x[score_key]), reverse=True)
            ranking = " > ".join(
                f"{r['style']}({f(r[score_key]):.3f})"
                for r in ordered
            )
            print(f"  {objective_name:20s}: {ranking}")

    print()
    print("[pairs]")
    for p in all_pairs:
        print(
            f"{p['terrain']:12s} "
            f"{p['objective_name']:20s} "
            f"{p['winner_style']:15s} > {p['loser_style']:15s} "
            f"margin={p['margin']:.3f} "
            f"conf={p['label_confidence']:.3f}"
        )


if __name__ == "__main__":
    main()
