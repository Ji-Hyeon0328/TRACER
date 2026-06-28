#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from statistics import mean, median


def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes"}


def deploy_score(row: dict) -> float:
    """
    Deployment-oriented score.
    Starts from quality_score, then penalizes invalid success/freshness/fallen spikes.
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
    ap.add_argument("--quality-csv", default="reports/tracer_world_style_sweep_3x3_v0_quality.csv")
    ap.add_argument("--out-agg", default="reports/tracer_world_style_sweep_3x3_v0_aggregate_by_style.csv")
    ap.add_argument("--out-pairs-csv", default="data/preference_datasets/tracer_world_style_preference_pairs_3x3_v0.csv")
    ap.add_argument("--out-pairs-jsonl", default="data/preference_datasets/tracer_world_style_preference_pairs_3x3_v0.jsonl")
    ap.add_argument("--min-margin", type=float, default=0.03)
    args = ap.parse_args()

    rows = []
    with open(args.quality_csv, newline="") as fp:
        for r in csv.DictReader(fp):
            r["quality_score"] = f(r.get("quality_score"))
            r["deploy_score"] = deploy_score(r)
            r["success_proxy"] = as_bool(r.get("success_proxy"))
            r["progress_ratio"] = f(r.get("progress_ratio"))
            r["distance_xy_proxy"] = f(r.get("distance_xy_proxy"))
            r["mpc_vx_mean"] = f(r.get("mpc_vx_mean"))
            r["ram_run_fallen_p90"] = f(r.get("ram_run_fallen_p90"))
            r["debug_fresh_rate_1p0s"] = f(r.get("debug_fresh_rate_1p0s"))
            rows.append(r)

    groups = {}
    for r in rows:
        groups.setdefault((r["terrain"], r["style"]), []).append(r)

    agg_rows = []
    for (terrain, style), ds in sorted(groups.items()):
        q = [r["quality_score"] for r in ds]
        dep = [r["deploy_score"] for r in ds]
        prog = [r["progress_ratio"] for r in ds]
        fresh = [r["debug_fresh_rate_1p0s"] for r in ds]
        fallen = [r["ram_run_fallen_p90"] for r in ds]
        dist = [r["distance_xy_proxy"] for r in ds]

        success_rate = sum(1 for r in ds if r["success_proxy"]) / max(1, len(ds))

        agg = {
            "terrain": terrain,
            "style": style,
            "n": len(ds),
            "success_rate": success_rate,
            "quality_mean": mean(q),
            "quality_median": median(q),
            "quality_min": min(q),
            "quality_max": max(q),
            "deploy_mean": mean(dep),
            "deploy_median": median(dep),
            "deploy_min": min(dep),
            "deploy_max": max(dep),
            "progress_mean": mean(prog),
            "progress_median": median(prog),
            "distance_mean": mean(dist),
            "fallen_p90_mean": mean(fallen),
            "fallen_p90_max": max(fallen),
            "fresh1_mean": mean(fresh),
            "fresh1_min": min(fresh),
            "episode_ids": ";".join(r["episode_id"] for r in ds),
        }
        agg_rows.append(agg)

    out_agg = Path(args.out_agg)
    out_agg.parent.mkdir(parents=True, exist_ok=True)

    agg_fields = [
        "terrain",
        "style",
        "n",
        "success_rate",
        "quality_mean",
        "quality_median",
        "quality_min",
        "quality_max",
        "deploy_mean",
        "deploy_median",
        "deploy_min",
        "deploy_max",
        "progress_mean",
        "progress_median",
        "distance_mean",
        "fallen_p90_mean",
        "fallen_p90_max",
        "fresh1_mean",
        "fresh1_min",
        "episode_ids",
    ]

    with out_agg.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=agg_fields)
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)

    by_terrain = {}
    for r in agg_rows:
        by_terrain.setdefault(r["terrain"], []).append(r)

    pair_rows = []
    for terrain, ds in sorted(by_terrain.items()):
        for a, b in combinations(ds, 2):
            qa = f(a["deploy_mean"])
            qb = f(b["deploy_mean"])
            margin = abs(qa - qb)

            if margin < args.min_margin:
                continue

            winner, loser = (a, b) if qa > qb else (b, a)
            pair = {
                "terrain": terrain,
                "winner_style": winner["style"],
                "loser_style": loser["style"],
                "winner_deploy_mean": f(winner["deploy_mean"]),
                "loser_deploy_mean": f(loser["deploy_mean"]),
                "winner_quality_mean": f(winner["quality_mean"]),
                "loser_quality_mean": f(loser["quality_mean"]),
                "winner_success_rate": f(winner["success_rate"]),
                "loser_success_rate": f(loser["success_rate"]),
                "winner_progress_mean": f(winner["progress_mean"]),
                "loser_progress_mean": f(loser["progress_mean"]),
                "winner_fallen_p90_max": f(winner["fallen_p90_max"]),
                "loser_fallen_p90_max": f(loser["fallen_p90_max"]),
                "winner_fresh1_min": f(winner["fresh1_min"]),
                "loser_fresh1_min": f(loser["fresh1_min"]),
                "margin_deploy_mean": margin,
                "winner_episode_ids": winner["episode_ids"],
                "loser_episode_ids": loser["episode_ids"],
                "label_source": "world_style_sweep_3x3_v0_aggregate_deploy_mean",
            }
            pair_rows.append(pair)

    out_pairs_csv = Path(args.out_pairs_csv)
    out_pairs_jsonl = Path(args.out_pairs_jsonl)
    out_pairs_csv.parent.mkdir(parents=True, exist_ok=True)

    pair_fields = [
        "terrain",
        "winner_style",
        "loser_style",
        "winner_deploy_mean",
        "loser_deploy_mean",
        "winner_quality_mean",
        "loser_quality_mean",
        "winner_success_rate",
        "loser_success_rate",
        "winner_progress_mean",
        "loser_progress_mean",
        "winner_fallen_p90_max",
        "loser_fallen_p90_max",
        "winner_fresh1_min",
        "loser_fresh1_min",
        "margin_deploy_mean",
        "winner_episode_ids",
        "loser_episode_ids",
        "label_source",
    ]

    with out_pairs_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=pair_fields)
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)

    with out_pairs_jsonl.open("w") as fp:
        for r in pair_rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[TRACER] input_rows={len(rows)} aggregate_rows={len(agg_rows)} pairs={len(pair_rows)}")
    print(f"[TRACER] wrote {out_agg}")
    print(f"[TRACER] wrote {out_pairs_csv}")
    print(f"[TRACER] wrote {out_pairs_jsonl}")
    print()

    for terrain, ds in sorted(by_terrain.items()):
        print(f"[{terrain}]")
        for r in sorted(ds, key=lambda x: x["deploy_mean"], reverse=True):
            print(
                f"  {r['style']:15s} "
                f"deploy={r['deploy_mean']:.3f} "
                f"q={r['quality_mean']:.3f} "
                f"succ={r['success_rate']:.3f} "
                f"progress={r['progress_mean']:.3f} "
                f"fallen_max={r['fallen_p90_max']:.3g} "
                f"fresh_min={r['fresh1_min']:.3f}"
            )

    print()
    print("[pairs]")
    for r in pair_rows:
        print(
            f"{r['terrain']:12s} "
            f"{r['winner_style']:15s} > {r['loser_style']:15s} "
            f"margin={r['margin_deploy_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
