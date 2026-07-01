#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PROFILE_BETA = {
    "balanced":          [0.34, 0.33, 0.33],
    "motion":            [0.65, 0.20, 0.15],
    "stability":         [0.20, 0.65, 0.15],
    "energy":            [0.20, 0.20, 0.60],
    "motion_extreme":    [0.85, 0.10, 0.05],
    "stability_extreme": [0.05, 0.90, 0.05],
    "energy_extreme":    [0.05, 0.10, 0.85],
}


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def parse_beta(v: Any, profile: str) -> list[float]:
    try:
        out = [float(x) for x in json.loads(str(v))]
        if len(out) >= 3:
            return out[:3]
    except Exception:
        pass
    return PROFILE_BETA.get(profile, PROFILE_BETA["balanced"])


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def get_col(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for n in names:
        if n in row:
            return to_float(row.get(n), default)
    return default


def classify_manifest_row(row: dict[str, Any]) -> str:
    stable = get_col(row, "stable_frac")
    margin = get_col(row, "target_height_margin_v2", default=-999.0)
    score = get_col(row, "R_profile_v2_gated_mean", "R_gated_mean")
    terrain = str(row.get("terrain", ""))

    # Slippery currently has no reliable positive under Phase-A action space.
    # Keep it as borderline/recovery unless future θ-action data proves otherwise.
    if terrain == "slippery_mild_flat":
        if stable >= 0.60 and margin >= -0.05:
            return "borderline_teacher"
        return "risk_recovery"

    if stable >= 0.99 and margin >= -0.03 and score >= 0.25:
        return "positive_teacher"

    if stable >= 0.50:
        return "borderline_teacher"

    return "risk_recovery"


def infer_targets(row: dict[str, Any], *, duration_sec: float) -> dict[str, float]:
    terrain = str(row.get("terrain", "unknown"))

    ref_vx = get_col(row, "ref_vx")
    ref_h = get_col(row, "ref_body_height", default=0.32)
    ref_c = get_col(row, "ref_swing_clearance", default=0.04)

    zmin = get_col(row, "zmin_mean", "proprio_base_z_min")
    below22 = get_col(row, "below22_mean", "proprio_base_z_below_0p22_frac")
    stable = get_col(row, "stable_frac")
    dist = get_col(row, "distance_mean", "distance_xy_proxy")
    roll = abs(get_col(row, "rollmax_mean", "proprio_roll_abs_max"))
    pitch = abs(get_col(row, "pitchmax_mean", "proprio_pitch_abs_max"))

    margin = get_col(row, "target_height_margin_v2", default=999.0)
    if margin == 999.0:
        expected_min_z = max(0.18, ref_h - 0.14)
        margin = zmin - expected_min_z
    else:
        expected_min_z = get_col(row, "expected_min_z_v2", default=max(0.18, ref_h - 0.14))

    score = get_col(row, "R_profile_v2_gated_mean", "R_gated_mean")

    expected_progress = max(0.03, abs(ref_vx) * duration_sec * 0.50)
    progress_ratio = clamp(dist / expected_progress)
    progress_mismatch = clamp(1.0 - progress_ratio)

    vertical_sink = clamp(max(0.0, -margin) / 0.12)
    orientation_risk = clamp(0.5 * min(1.0, roll / 0.50) + 0.5 * min(1.0, pitch / 0.50))
    low_height_risk = clamp(max(below22, vertical_sink))
    instability = clamp(1.0 - stable)

    recovery_needed = 1.0 if (
        stable < 0.50
        or vertical_sink > 0.50
        or orientation_risk > 0.75
        or score < 0.15
    ) else 0.0

    # Uncertainty proxy: high when near border, unstable across repeats, or terrain is slippery.
    border_uncertainty = 1.0 - abs(stable - 0.5) * 2.0
    terrain_uncertainty = 0.25 if terrain == "slippery_mild_flat" else 0.0
    uncertainty_proxy = clamp(0.55 * border_uncertainty + 0.25 * instability + 0.20 * terrain_uncertainty)

    return {
        "a_hl_vx": ref_vx,
        "a_hl_yaw_rate": get_col(row, "ref_yaw_rate"),
        "a_hl_body_height": ref_h,
        "a_hl_swing_clearance": ref_c,

        "ram_vertical_sink_target": vertical_sink,
        "ram_progress_mismatch_target": progress_mismatch,
        "ram_orientation_risk_target": orientation_risk,
        "ram_low_height_risk_target": low_height_risk,
        "ram_recovery_needed_target": recovery_needed,
        "ram_uncertainty_proxy_target": uncertainty_proxy,

        "objective_preference_score": score,
        "meta_planner_teacher_score": score,
        "meta_planner_usable_teacher": 1.0 if score >= 0.25 and stable >= 0.50 else 0.0,

        "expected_min_z": expected_min_z,
        "target_height_margin": margin,
        "progress_ratio": progress_ratio,
    }


def find_profile_csv(run_dir: Path) -> Path:
    v2 = run_dir / "reference_sweep_profile_rescore_phase_a_v2_target_height.csv"
    if v2.exists():
        return v2

    v0 = run_dir / "reference_sweep_profile_rescore_phase_a_v0.csv"
    if v0.exists():
        return v0

    raise FileNotFoundError(f"No profile rescore csv found in {run_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-dir", default="data/phase_a_training_manifest_v0")
    ap.add_argument("--duration-sec", type=float, default=5.0)
    args = ap.parse_args()

    rows_out: list[dict[str, Any]] = []

    for run_str in args.runs:
        run_dir = Path(run_str)
        profile_csv = find_profile_csv(run_dir)
        rows = read_csv(profile_csv)

        for r in rows:
            profile = str(r.get("profile", "balanced"))
            beta = parse_beta(r.get("beta", ""), profile)
            label = classify_manifest_row(r)
            targets = infer_targets(r, duration_sec=args.duration_sec)

            out: dict[str, Any] = {
                "source_run_dir": str(run_dir),
                "source_profile_csv": str(profile_csv),
                "terrain": r.get("terrain", ""),
                "actual_world_name": r.get("actual_world_name", ""),
                "world_check_ok": r.get("world_check_ok", ""),
                "profile": profile,
                "beta_motion": beta[0],
                "beta_stability": beta[1],
                "beta_energy": beta[2],
                "preset": r.get("preset", ""),
                "manifest_label": label,

                "stable_frac": get_col(r, "stable_frac"),
                "valid_frac": get_col(r, "valid_frac", default=1.0),
                "zmin_mean": get_col(r, "zmin_mean", "proprio_base_z_min"),
                "below22_mean": get_col(r, "below22_mean", "proprio_base_z_below_0p22_frac"),
                "distance_mean": get_col(r, "distance_mean", "distance_xy_proxy"),
                "rollmax_mean": get_col(r, "rollmax_mean", "proprio_roll_abs_max"),
                "pitchmax_mean": get_col(r, "pitchmax_mean", "proprio_pitch_abs_max"),
                "R_score": get_col(r, "R_profile_v2_gated_mean", "R_gated_mean"),
            }
            out.update(targets)
            rows_out.append(out)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "phase_a_training_manifest_v0.csv"
    jsonl_path = out_dir / "phase_a_training_manifest_v0.jsonl"
    summary_path = out_dir / "phase_a_training_manifest_summary_v0.json"

    write_csv(csv_path, rows_out)
    write_jsonl(jsonl_path, rows_out)

    counts: dict[str, int] = {}
    terrain_counts: dict[str, int] = {}
    for r in rows_out:
        label = str(r["manifest_label"])
        terrain = str(r["terrain"])
        counts[label] = counts.get(label, 0) + 1
        terrain_counts[f"{terrain}::{label}"] = terrain_counts.get(f"{terrain}::{label}", 0) + 1

    summary = {
        "num_rows": len(rows_out),
        "runs": args.runs,
        "counts": counts,
        "terrain_counts": terrain_counts,
        "outputs": {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
            "summary": str(summary_path),
        },
        "note": (
            "Phase-A manifest for TRACER RAM teacher-student, bootstrapped objective "
            "learning, and meta-gait planner warm-start. Slippery is intentionally "
            "kept as borderline/risk unless reliable positive data emerges."
        ),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print()
    print("===== examples by terrain/label =====")

    for terrain in sorted(set(str(r["terrain"]) for r in rows_out)):
        print()
        print(f"terrain={terrain}")
        subset = [r for r in rows_out if str(r["terrain"]) == terrain]
        subset.sort(key=lambda r: float(r["R_score"]), reverse=True)
        for r in subset[:12]:
            print(
                f"  {r['manifest_label']:18s} "
                f"{r['profile']:18s} "
                f"{r['preset']:44s} "
                f"score={float(r['R_score']):.3f} "
                f"stable={float(r['stable_frac']):.2f} "
                f"sink={float(r['ram_vertical_sink_target']):.2f} "
                f"mismatch={float(r['ram_progress_mismatch_target']):.2f} "
                f"unc={float(r['ram_uncertainty_proxy_target']):.2f} "
                f"a=[{float(r['a_hl_vx']):.3f},"
                f"{float(r['a_hl_body_height']):.3f},"
                f"{float(r['a_hl_swing_clearance']):.3f}]"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
