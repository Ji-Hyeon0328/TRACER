#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROFILE_KEYS = [
    "balanced",
    "motion",
    "stability",
    "energy",
    "motion_extreme",
    "stability_extreme",
    "energy_extreme",
]


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def to_bool(v: Any) -> bool:
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write")

    fields: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def frac_true(xs: list[bool]) -> float:
    if not xs:
        return 0.0
    return sum(1 for x in xs if x) / len(xs)


def key_for_row(r: dict[str, Any]) -> tuple:
    return (
        str(r.get("terrain", "")),
        round(to_float(r.get("ref_vx")), 6),
        round(to_float(r.get("ref_yaw_rate")), 6),
        round(to_float(r.get("ref_body_height")), 6),
        round(to_float(r.get("ref_swing_clearance")), 6),
        round(to_float(r.get("ref_enable"), 1.0), 6),
    )


def row_metric(r: dict[str, Any], *names: str) -> float:
    for n in names:
        if n in r:
            return to_float(r.get(n))
    return 0.0


def infer_stable(r: dict[str, Any]) -> bool:
    if "proprio_base_height_stable" in r:
        return to_bool(r.get("proprio_base_height_stable"))

    zmin = row_metric(r, "proprio_base_z_min", "zmin")
    below22 = row_metric(r, "proprio_base_z_below_0p22_frac", "below22")
    roll = row_metric(r, "proprio_roll_abs_max", "rollmax")
    pitch = row_metric(r, "proprio_pitch_abs_max", "pitchmax")

    return zmin >= 0.18 and below22 <= 0.35 and roll <= 0.70 and pitch <= 0.70


def classify(stable_prob: float, zmin_mean: float, below22_mean: float, uncertainty: float) -> str:
    if stable_prob >= 0.80 and zmin_mean >= 0.22 and below22_mean <= 0.10 and uncertainty <= 0.30:
        return "robust_positive"

    if stable_prob >= 0.60 and zmin_mean >= 0.18 and below22_mean <= 0.35:
        return "probabilistic_positive"

    if stable_prob >= 0.35:
        return "borderline_high_variance"

    return "risk_negative"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-dir", default="data/phase_a_action_outcome_aggregates_v0")
    args = ap.parse_args()

    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)

    for run_str in args.runs:
        run = Path(run_str)
        p = run / "reference_sweep_scored_phase_a_v0.csv"
        if not p.exists():
            print(f"[WARN] missing scored csv: {p}")
            continue

        rows = read_csv(p)
        for r in rows:
            rr = dict(r)
            rr["source_run_dir"] = str(run)
            groups[key_for_row(rr)].append(rr)

    out_rows: list[dict[str, Any]] = []

    for key, rows in groups.items():
        terrain, vx, yaw, h, c, enable = key

        valid = [to_bool(r.get("valid_data", "false")) for r in rows]
        stable = [infer_stable(r) for r in rows]

        zmins = [row_metric(r, "proprio_base_z_min", "zmin") for r in rows]
        below22s = [row_metric(r, "proprio_base_z_below_0p22_frac", "below22") for r in rows]
        rolls = [row_metric(r, "proprio_roll_abs_max", "rollmax") for r in rows]
        pitches = [row_metric(r, "proprio_pitch_abs_max", "pitchmax") for r in rows]
        dists = [row_metric(r, "distance_xy_proxy", "distance") for r in rows]

        stable_prob = frac_true(stable)
        valid_prob = frac_true(valid)

        # Bernoulli uncertainty plus continuous outcome spread.
        bernoulli_unc = 4.0 * stable_prob * (1.0 - stable_prob)
        z_unc = min(1.0, std(zmins) / 0.10)
        below_unc = min(1.0, std(below22s) / 0.30)
        outcome_unc = min(1.0, 0.50 * bernoulli_unc + 0.25 * z_unc + 0.25 * below_unc)

        row: dict[str, Any] = {
            "terrain": terrain,
            "ref_vx": vx,
            "ref_yaw_rate": yaw,
            "ref_body_height": h,
            "ref_swing_clearance": c,
            "ref_enable": enable,
            "n_episodes": len(rows),
            "valid_prob": valid_prob,
            "stable_prob": stable_prob,
            "zmin_mean": mean(zmins),
            "zmin_std": std(zmins),
            "below22_mean": mean(below22s),
            "below22_std": std(below22s),
            "rollmax_mean": mean(rolls),
            "rollmax_std": std(rolls),
            "pitchmax_mean": mean(pitches),
            "pitchmax_std": std(pitches),
            "distance_mean": mean(dists),
            "distance_std": std(dists),
            "outcome_uncertainty": outcome_unc,
            "source_runs_json": json.dumps(sorted(set(r["source_run_dir"] for r in rows))),
            "source_presets_json": json.dumps(sorted(set(str(r.get("preset", "")) for r in rows))),
        }

        for profile in PROFILE_KEYS:
            col = f"R_gated_{profile}"
            vals = [to_float(r.get(col)) for r in rows if col in r]
            row[f"{col}_mean"] = mean(vals)
            row[f"{col}_std"] = std(vals)

        row["outcome_label"] = classify(
            stable_prob=stable_prob,
            zmin_mean=row["zmin_mean"],
            below22_mean=row["below22_mean"],
            uncertainty=outcome_unc,
        )

        out_rows.append(row)

    out_rows.sort(
        key=lambda r: (
            str(r["terrain"]),
            -float(r["stable_prob"]),
            float(r["outcome_uncertainty"]),
            -float(r["zmin_mean"]),
        )
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "phase_a_action_outcome_aggregates_v0.csv"
    summary_path = out_dir / "phase_a_action_outcome_aggregates_summary_v0.json"

    write_csv(csv_path, out_rows)

    counts: dict[str, int] = {}
    terrain_counts: dict[str, int] = {}
    for r in out_rows:
        label = str(r["outcome_label"])
        terrain = str(r["terrain"])
        counts[label] = counts.get(label, 0) + 1
        terrain_counts[f"{terrain}::{label}"] = terrain_counts.get(f"{terrain}::{label}", 0) + 1

    summary = {
        "num_groups": len(out_rows),
        "runs": args.runs,
        "counts": counts,
        "terrain_counts": terrain_counts,
        "outputs": {
            "csv": str(csv_path),
            "summary": str(summary_path),
        },
        "note": (
            "Aggregates same-action Phase-A rollout outcomes across runs. "
            "Use stable_prob and outcome_uncertainty instead of single-run labels."
        ),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print()
    print("===== aggregated outcomes =====")
    for r in out_rows:
        print(
            f"terrain={r['terrain']:20s} "
            f"label={r['outcome_label']:25s} "
            f"n={int(r['n_episodes']):3d} "
            f"stable_p={float(r['stable_prob']):.2f} "
            f"unc={float(r['outcome_uncertainty']):.2f} "
            f"z={float(r['zmin_mean']):.3f}±{float(r['zmin_std']):.3f} "
            f"below22={float(r['below22_mean']):.3f}±{float(r['below22_std']):.3f} "
            f"a=[{float(r['ref_vx']):.3f},{float(r['ref_body_height']):.3f},{float(r['ref_swing_clearance']):.3f}]"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
