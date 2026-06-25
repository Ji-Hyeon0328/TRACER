#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


FEATURE_KEYS = [
    "mpc_vx_mean",
    "mpc_yaw_mean",
    "mpc_body_height_mean",
    "mpc_clearance_mean",
    "mpc_enable_mean",
    "beta_motion_mean",
    "beta_stability_mean",
    "beta_energy_mean",
    "dx",
    "dy",
    "dz",
    "min_z",
    "max_z",
    "max_abs_roll",
    "max_abs_pitch",
    "observed_vx",
    "observed_vy",
    "abs_dx",
    "abs_dy",
    "z_drop",
    "orientation_peak",
    "tracking_vx_error",
    "trajectory_cost_v0",
]


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y"}


def parse_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def load_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for raw in csv.DictReader(f):
            row: dict[str, Any] = dict(raw)

            for k in FEATURE_KEYS:
                if k in row:
                    row[k] = parse_float(row[k])

            if "fall_like" in row:
                row["fall_like"] = parse_bool(row["fall_like"])

            if "num_samples" in row:
                row["num_samples"] = int(parse_float(row["num_samples"], 0.0))

            if "proprio_finite_rows" in row:
                row["proprio_finite_rows"] = int(parse_float(row["proprio_finite_rows"], 0.0))

            if "mpc_finite_rows" in row:
                row["mpc_finite_rows"] = int(parse_float(row["mpc_finite_rows"], 0.0))

            rows.append(row)

    return rows


def compact_metrics(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "npz_path": row.get("npz_path"),
        "tag": row.get("tag"),
        "pose_source": row.get("pose_source"),
        "fall_like": bool(row.get("fall_like")),
    }
    for k in FEATURE_KEYS:
        if k in row:
            out[k] = row[k]
    return out


def reason_for_pair(w: dict[str, Any], l: dict[str, Any]) -> str:
    wf = bool(w.get("fall_like"))
    lf = bool(l.get("fall_like"))

    if not wf and lf:
        return "winner_did_not_fall_loser_fell"
    if wf and lf:
        return "both_fell_but_winner_has_lower_cost"
    if not wf and not lf:
        return "both_stable_but_winner_has_lower_cost"
    return "lower_cost"


def pair_confidence(gap: float, winner: dict[str, Any], loser: dict[str, Any], scale: float) -> float:
    wf = bool(winner.get("fall_like"))
    lf = bool(loser.get("fall_like"))

    if not wf and lf:
        return 1.0

    if gap <= 0.0:
        return 0.0

    return max(0.05, min(1.0, gap / max(scale, 1e-9)))


def build_pairs(
    rows: list[dict[str, Any]],
    *,
    min_margin: float,
    confidence_scale: float,
    max_pairs: int,
) -> list[dict[str, Any]]:
    valid = [
        r for r in rows
        if math.isfinite(parse_float(r.get("trajectory_cost_v0")))
        and str(r.get("pose_source", "none")) != "none"
    ]

    valid = sorted(valid, key=lambda r: parse_float(r["trajectory_cost_v0"]))

    pairs: list[dict[str, Any]] = []
    for i, winner in enumerate(valid):
        c_w = parse_float(winner["trajectory_cost_v0"])
        for j in range(i + 1, len(valid)):
            loser = valid[j]
            c_l = parse_float(loser["trajectory_cost_v0"])
            gap = c_l - c_w

            if gap < min_margin:
                continue

            pair = {
                "pair_id": f"pref_{len(pairs):06d}",
                "winner_npz": winner.get("npz_path"),
                "loser_npz": loser.get("npz_path"),
                "winner_tag": winner.get("tag"),
                "loser_tag": loser.get("tag"),
                "winner_cost": c_w,
                "loser_cost": c_l,
                "cost_gap": gap,
                "confidence": pair_confidence(gap, winner, loser, confidence_scale),
                "reason": reason_for_pair(winner, loser),
                "winner": compact_metrics(winner),
                "loser": compact_metrics(loser),
            }
            pairs.append(pair)

            if max_pairs > 0 and len(pairs) >= max_pairs:
                return pairs

    return pairs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_ranking_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ranking = sorted(rows, key=lambda r: parse_float(r.get("trajectory_cost_v0")))

    fields = [
        "rank",
        "npz_path",
        "tag",
        "pose_source",
        "fall_like",
        "trajectory_cost_v0",
        "dx",
        "dy",
        "min_z",
        "max_abs_roll",
        "max_abs_pitch",
        "mpc_vx_mean",
        "mpc_body_height_mean",
        "mpc_clearance_mean",
        "mpc_enable_mean",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for rank, row in enumerate(ranking, start=1):
            out = {"rank": rank}
            for k in fields:
                if k != "rank":
                    out[k] = row.get(k)
            writer.writerow(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics-csv",
        default="data/rollout_metrics/tracer_rollout_metrics_v0.csv",
    )
    ap.add_argument(
        "--out-dir",
        default="data/preference_datasets",
    )
    ap.add_argument(
        "--min-margin",
        type=float,
        default=0.25,
        help="Minimum trajectory_cost_v0 gap required to create a preference pair.",
    )
    ap.add_argument(
        "--confidence-scale",
        type=float,
        default=2.0,
        help="Cost gap that maps to confidence 1.0 for non-fall-difference pairs.",
    )
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Maximum pairs to write. 0 means no limit.",
    )
    args = ap.parse_args()

    metrics_csv = Path(args.metrics_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metrics(metrics_csv)
    pairs = build_pairs(
        rows,
        min_margin=args.min_margin,
        confidence_scale=args.confidence_scale,
        max_pairs=args.max_pairs,
    )

    pair_jsonl = out_dir / "tracer_rollout_preference_pairs_v0.jsonl"
    ranking_csv = out_dir / "tracer_rollout_ranking_v0.csv"

    write_jsonl(pair_jsonl, pairs)
    write_ranking_csv(ranking_csv, rows)

    ranking = sorted(rows, key=lambda r: parse_float(r.get("trajectory_cost_v0")))

    print(f"[TRACER] loaded metrics: {len(rows)}")
    print(f"[TRACER] wrote pairs:    {len(pairs)} -> {pair_jsonl}")
    print(f"[TRACER] wrote ranking:  {ranking_csv}")

    print("\n[TRACER] top ranking:")
    for rank, r in enumerate(ranking[:10], start=1):
        print(
            f"{rank:02d} "
            f"cost={parse_float(r.get('trajectory_cost_v0')):.3f} "
            f"fall={bool(r.get('fall_like'))} "
            f"dx={parse_float(r.get('dx')):.3f} "
            f"dy={parse_float(r.get('dy')):.3f} "
            f"min_z={parse_float(r.get('min_z')):.3f} "
            f"vx={parse_float(r.get('mpc_vx_mean')):.4f} "
            f"h={parse_float(r.get('mpc_body_height_mean')):.3f} "
            f"clr={parse_float(r.get('mpc_clearance_mean')):.3f} "
            f"{Path(str(r.get('npz_path'))).name}"
        )

    print("\n[TRACER] sample pairs:")
    for pair in pairs[:10]:
        print(
            f"{pair['pair_id']} "
            f"winner_cost={pair['winner_cost']:.3f} "
            f"loser_cost={pair['loser_cost']:.3f} "
            f"gap={pair['cost_gap']:.3f} "
            f"conf={pair['confidence']:.2f} "
            f"reason={pair['reason']}"
        )


if __name__ == "__main__":
    main()
