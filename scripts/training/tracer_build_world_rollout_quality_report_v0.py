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


def infer_world_id(policy_id: str) -> str:
    token = "_world_"
    if token not in policy_id:
        return ""
    return policy_id.split(token, 1)[1]


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

    # Cap progress ratio so a single odd overshoot does not dominate preference labels.
    progress_score = clamp(progress_ratio, 0.0, 1.0)

    # Risk penalties. Current RAM recovery head is the active gate; fallen is used
    # as a soft quality penalty only.
    risk_penalty = 0.45 * clamp(fallen_p90, 0.0, 1.0) + 0.35 * clamp(recovery, 0.0, 1.0)
    stale_penalty = 0.20 * clamp(1.0 - fresh1, 0.0, 1.0)

    # Success proxy is useful but should not dominate; it only adds a small bonus.
    success_bonus = 0.10 if bool(d.get("success_proxy")) else 0.0

    q = progress_score + success_bonus - risk_penalty - stale_penalty
    q = clamp(q, -1.0, 1.1)

    return {
        "world_id": infer_world_id(str(d.get("policy_id", ""))),
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
    ap.add_argument("--summary-glob", default="data/rollout_dataset_v0/summaries/*world_reload_robust3x3_v0*.json")
    ap.add_argument("--out-csv", default="reports/tracer_world_reload_robust3x3_v0_quality.csv")
    ap.add_argument("--out-agg", default="reports/tracer_world_reload_robust3x3_v0_quality_by_terrain.csv")
    args = ap.parse_args()

    files = sorted(Path(".").glob(args.summary_glob))
    rows = []

    for p in files:
        d = json.loads(p.read_text())
        q = quality_score(d)
        row = {
            "episode_id": d.get("episode_id", ""),
            "terrain": d.get("terrain", ""),
            "world_id": q["world_id"],
            "policy_id": d.get("policy_id", ""),
            "duration_sec": f(d.get("duration_sec")),
            "mpc_vx_mean": f(d.get("mpc_vx_mean")),
            "mpc_body_height_mean": f(d.get("mpc_body_height_mean")),
            "mpc_clearance_mean": f(d.get("mpc_clearance_mean")),
            "distance_xy_proxy": f(d.get("distance_xy_proxy")),
            "expected_distance": q["expected_distance"],
            "progress_ratio": q["progress_ratio"],
            "progress_score": q["progress_score"],
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
        "world_id",
        "policy_id",
        "duration_sec",
        "mpc_vx_mean",
        "mpc_body_height_mean",
        "mpc_clearance_mean",
        "distance_xy_proxy",
        "expected_distance",
        "progress_ratio",
        "progress_score",
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

    by_terrain = {}
    for r in rows:
        by_terrain.setdefault(r["terrain"], []).append(r)

    agg_rows = []
    for terrain, ds in sorted(by_terrain.items()):
        agg_rows.append({
            "terrain": terrain,
            "n": len(ds),
            "success_rate": sum(1 for r in ds if r["success_proxy"]) / max(1, len(ds)),
            "distance_mean": mean(r["distance_xy_proxy"] for r in ds),
            "expected_distance_mean": mean(r["expected_distance"] for r in ds),
            "progress_ratio_mean": mean(r["progress_ratio"] for r in ds),
            "fallen_p90_mean": mean(r["ram_run_fallen_p90"] for r in ds),
            "fresh1_mean": mean(r["debug_fresh_rate_1p0s"] for r in ds),
            "quality_score_mean": mean(r["quality_score"] for r in ds),
            "quality_score_min": min(r["quality_score"] for r in ds),
            "quality_score_max": max(r["quality_score"] for r in ds),
        })

    out_agg = Path(args.out_agg)
    agg_fields = [
        "terrain",
        "n",
        "success_rate",
        "distance_mean",
        "expected_distance_mean",
        "progress_ratio_mean",
        "fallen_p90_mean",
        "fresh1_mean",
        "quality_score_mean",
        "quality_score_min",
        "quality_score_max",
    ]

    with out_agg.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=agg_fields)
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)

    print(f"[TRACER] summaries={len(files)}")
    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_agg}")
    print()
    for r in agg_rows:
        print(
            f"{r['terrain']:12s} "
            f"n={r['n']} "
            f"succ={r['success_rate']:.3f} "
            f"progress={r['progress_ratio_mean']:.3f} "
            f"quality={r['quality_score_mean']:.3f} "
            f"fallen={r['fallen_p90_mean']:.3g} "
            f"fresh={r['fresh1_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
