#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def f(x, default=0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_style_from_policy_id(policy_id: str, terrain: str) -> str:
    # Examples:
    # world_style_sweep_smoke_v1_flat_normal_fast_world_stairs_single
    # world_style_sweep_smoke_v1_rough_mid_high_clearance_world_tracer_rough_mid
    prefix = "world_style_sweep_smoke_v1_"
    if prefix in policy_id:
        tail = policy_id.split(prefix, 1)[1]
        marker = "_world_"
        if marker in tail:
            tail = tail.split(marker, 1)[0]

        terrain_prefix = terrain + "_"
        if tail.startswith(terrain_prefix):
            return tail[len(terrain_prefix):]

    for style in ["high_clearance", "cautious", "fast", "conservative"]:
        if f"_{style}_" in policy_id or policy_id.endswith("_" + style):
            return style

    return "unknown"


def quality_score(d: dict) -> dict:
    duration = f(d.get("duration_sec"))
    vx = f(d.get("mpc_vx_mean"))
    enable = f(d.get("mpc_enable_mean"), 1.0)
    dist = f(d.get("distance_xy_proxy"))
    fallen_p90 = f(d.get("ram_run_fallen_p90"))
    recovery = f(d.get("ram_recovery_needed_mean"))
    fresh1 = f(d.get("debug_fresh_rate_1p0s"))

    expected = abs(vx) * duration * enable
    progress_ratio = dist / max(expected, 1e-6)

    progress_score = clamp(progress_ratio, 0.0, 1.0)
    risk_penalty = 0.45 * clamp(fallen_p90, 0.0, 1.0) + 0.35 * clamp(recovery, 0.0, 1.0)
    stale_penalty = 0.20 * clamp(1.0 - fresh1, 0.0, 1.0)
    success_bonus = 0.10 if bool(d.get("success_proxy")) else 0.0

    q = progress_score + success_bonus - risk_penalty - stale_penalty
    q = clamp(q, -1.0, 1.1)

    return {
        "expected_distance": expected,
        "progress_ratio": progress_ratio,
        "progress_score": progress_score,
        "risk_penalty": risk_penalty,
        "stale_penalty": stale_penalty,
        "success_bonus": success_bonus,
        "quality_score": q,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-glob", default="data/rollout_dataset_v0/summaries/*world_style_sweep_smoke_v1*.json")
    ap.add_argument("--out-csv", default="reports/tracer_world_style_sweep_smoke_v1_quality.csv")
    ap.add_argument("--out-ranking", default="reports/tracer_world_style_sweep_smoke_v1_ranking_by_terrain.csv")
    args = ap.parse_args()

    files = sorted(Path(".").glob(args.summary_glob))
    rows = []

    for p in files:
        d = json.loads(p.read_text())
        terrain = str(d.get("terrain", "unknown"))
        policy_id = str(d.get("policy_id", ""))
        style = parse_style_from_policy_id(policy_id, terrain)
        q = quality_score(d)

        row = {
            "episode_id": d.get("episode_id", ""),
            "terrain": terrain,
            "style": style,
            "policy_id": policy_id,
            "duration_sec": f(d.get("duration_sec")),
            "mpc_vx_mean": f(d.get("mpc_vx_mean")),
            "mpc_enable_mean": f(d.get("mpc_enable_mean")),
            "distance_xy_proxy": f(d.get("distance_xy_proxy")),
            "expected_distance": q["expected_distance"],
            "progress_ratio": q["progress_ratio"],
            "success_proxy": bool(d.get("success_proxy")),
            "ram_run_fallen_p90": f(d.get("ram_run_fallen_p90")),
            "ram_recovery_needed_mean": f(d.get("ram_recovery_needed_mean")),
            "debug_fresh_rate_1p0s": f(d.get("debug_fresh_rate_1p0s")),
            "risk_penalty": q["risk_penalty"],
            "stale_penalty": q["stale_penalty"],
            "success_bonus": q["success_bonus"],
            "quality_score": q["quality_score"],
            "summary_json": str(p),
            "step_csv": d.get("step_csv", ""),
        }
        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "episode_id",
        "terrain",
        "style",
        "policy_id",
        "duration_sec",
        "mpc_vx_mean",
        "mpc_enable_mean",
        "distance_xy_proxy",
        "expected_distance",
        "progress_ratio",
        "success_proxy",
        "ram_run_fallen_p90",
        "ram_recovery_needed_mean",
        "debug_fresh_rate_1p0s",
        "risk_penalty",
        "stale_penalty",
        "success_bonus",
        "quality_score",
        "summary_json",
        "step_csv",
    ]

    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ranking_rows = []
    by_terrain = {}
    for r in rows:
        by_terrain.setdefault(r["terrain"], []).append(r)

    for terrain, ds in sorted(by_terrain.items()):
        ds_sorted = sorted(ds, key=lambda x: x["quality_score"], reverse=True)
        for rank, r in enumerate(ds_sorted, start=1):
            ranking_rows.append({
                "terrain": terrain,
                "rank": rank,
                "style": r["style"],
                "quality_score": r["quality_score"],
                "success_proxy": r["success_proxy"],
                "progress_ratio": r["progress_ratio"],
                "distance_xy_proxy": r["distance_xy_proxy"],
                "mpc_vx_mean": r["mpc_vx_mean"],
                "fallen_p90": r["ram_run_fallen_p90"],
                "fresh1": r["debug_fresh_rate_1p0s"],
                "episode_id": r["episode_id"],
            })

    out_rank = Path(args.out_ranking)
    rank_fields = [
        "terrain",
        "rank",
        "style",
        "quality_score",
        "success_proxy",
        "progress_ratio",
        "distance_xy_proxy",
        "mpc_vx_mean",
        "fallen_p90",
        "fresh1",
        "episode_id",
    ]

    with out_rank.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=rank_fields)
        w.writeheader()
        for r in ranking_rows:
            w.writerow(r)

    print(f"[TRACER] summaries={len(rows)}")
    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_rank}")
    print()

    for terrain, ds in sorted(by_terrain.items()):
        print(f"[{terrain}]")
        for r in sorted(ds, key=lambda x: x["quality_score"], reverse=True):
            print(
                f"  {r['style']:15s} "
                f"q={r['quality_score']:.3f} "
                f"succ={r['success_proxy']} "
                f"progress={r['progress_ratio']:.3f} "
                f"dist={r['distance_xy_proxy']:.3f} "
                f"fallen={r['ram_run_fallen_p90']:.3g} "
                f"fresh={r['debug_fresh_rate_1p0s']:.3f}"
            )


if __name__ == "__main__":
    main()
