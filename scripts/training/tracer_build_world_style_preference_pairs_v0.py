#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path


def as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes"}


def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def deploy_score(row: dict) -> float:
    """
    Deployment-oriented score.
    Stronger penalty for invalid/freshness failure than the progress-only quality score.
    """
    q = f(row.get("quality_score"))
    fresh = f(row.get("debug_fresh_rate_1p0s"))
    success = as_bool(row.get("success_proxy"))
    fallen = f(row.get("ram_run_fallen_p90"))

    score = q
    if not success:
        score -= 0.25
    if fresh < 0.90:
        score -= 0.30 * (0.90 - fresh) / 0.90
    if fallen > 0.30:
        score -= 0.50 * min(1.0, fallen)

    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quality-csv",
        default="reports/tracer_world_style_sweep_smoke_v1_quality.csv",
    )
    ap.add_argument(
        "--out-jsonl",
        default="data/preference_datasets/tracer_world_style_preference_pairs_v0.jsonl",
    )
    ap.add_argument(
        "--out-csv",
        default="data/preference_datasets/tracer_world_style_preference_pairs_v0.csv",
    )
    ap.add_argument("--min-margin", type=float, default=0.03)
    args = ap.parse_args()

    rows = []
    with open(args.quality_csv, newline="") as fp:
        for r in csv.DictReader(fp):
            r["quality_score_progress"] = f(r.get("quality_score"))
            r["quality_score_deploy"] = deploy_score(r)
            rows.append(r)

    by_terrain = {}
    for r in rows:
        by_terrain.setdefault(r["terrain"], []).append(r)

    pairs = []
    for terrain, ds in sorted(by_terrain.items()):
        for a, b in combinations(ds, 2):
            qa = f(a["quality_score_deploy"])
            qb = f(b["quality_score_deploy"])
            margin = abs(qa - qb)

            if margin < args.min_margin:
                continue

            winner, loser = (a, b) if qa > qb else (b, a)

            pair = {
                "terrain": terrain,
                "winner_style": winner["style"],
                "loser_style": loser["style"],
                "winner_episode_id": winner["episode_id"],
                "loser_episode_id": loser["episode_id"],
                "winner_quality_deploy": f(winner["quality_score_deploy"]),
                "loser_quality_deploy": f(loser["quality_score_deploy"]),
                "winner_quality_progress": f(winner["quality_score_progress"]),
                "loser_quality_progress": f(loser["quality_score_progress"]),
                "margin_deploy": margin,
                "winner_success": as_bool(winner["success_proxy"]),
                "loser_success": as_bool(loser["success_proxy"]),
                "winner_progress_ratio": f(winner["progress_ratio"]),
                "loser_progress_ratio": f(loser["progress_ratio"]),
                "winner_fresh1": f(winner["debug_fresh_rate_1p0s"]),
                "loser_fresh1": f(loser["debug_fresh_rate_1p0s"]),
                "winner_fallen_p90": f(winner["ram_run_fallen_p90"]),
                "loser_fallen_p90": f(loser["ram_run_fallen_p90"]),
                "label_source": "world_style_sweep_smoke_v1_deploy_score",
            }
            pairs.append(pair)

    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w") as fp:
        for p in pairs:
            fp.write(json.dumps(p, ensure_ascii=False) + "\n")

    fields = [
        "terrain",
        "winner_style",
        "loser_style",
        "winner_quality_deploy",
        "loser_quality_deploy",
        "winner_quality_progress",
        "loser_quality_progress",
        "margin_deploy",
        "winner_success",
        "loser_success",
        "winner_progress_ratio",
        "loser_progress_ratio",
        "winner_fresh1",
        "loser_fresh1",
        "winner_fallen_p90",
        "loser_fallen_p90",
        "winner_episode_id",
        "loser_episode_id",
        "label_source",
    ]

    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for p in pairs:
            w.writerow({k: p.get(k, "") for k in fields})

    print(f"[TRACER] rows={len(rows)} pairs={len(pairs)}")
    print(f"[TRACER] wrote {out_jsonl}")
    print(f"[TRACER] wrote {out_csv}")
    print()

    for p in pairs:
        print(
            f"{p['terrain']:12s} "
            f"{p['winner_style']:15s} > {p['loser_style']:15s} "
            f"margin={p['margin_deploy']:.3f} "
            f"deploy=({p['winner_quality_deploy']:.3f},{p['loser_quality_deploy']:.3f})"
        )


if __name__ == "__main__":
    main()
