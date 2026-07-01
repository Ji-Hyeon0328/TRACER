#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def parse_beta(s: Any) -> list[float]:
    if isinstance(s, list):
        return [float(x) for x in s]
    try:
        return [float(x) for x in json.loads(str(s))]
    except Exception:
        return [0.34, 0.33, 0.33]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write")

    fieldnames: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def get_col(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for n in names:
        if n in row:
            return to_float(row.get(n), default)
    return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--max-command-vx", type=float, default=0.12)
    ap.add_argument("--min-absolute-z", type=float, default=0.18)
    ap.add_argument("--allowed-body-drop", type=float, default=0.14)
    ap.add_argument("--height-soft-margin", type=float, default=0.08)
    ap.add_argument("--aux-penalty", type=float, default=1.25)
    args = ap.parse_args()

    rows = read_csv(Path(args.input_csv))
    if not rows:
        raise RuntimeError(f"No rows: {args.input_csv}")

    max_dist = max(1e-6, max(get_col(r, "distance_mean", "distance_xy_proxy") for r in rows))
    max_vx = max(1e-6, args.max_command_vx)

    out: list[dict[str, Any]] = []

    for r in rows:
        rr = dict(r)

        beta = parse_beta(rr.get("beta", ""))
        b_motion, b_stability, b_energy = beta[0], beta[1], beta[2]

        ref_vx = get_col(rr, "ref_vx", "vx", "cmd_vx")
        ref_h = get_col(rr, "ref_body_height", "body_height", "cmd_body_height", default=0.32)
        ref_c = get_col(rr, "ref_swing_clearance", "swing_clearance", "cmd_swing_clearance", default=0.04)

        zmin = get_col(rr, "zmin_mean", "proprio_base_z_min")
        below22 = get_col(rr, "below22_mean", "proprio_base_z_below_0p22_frac")
        dist = get_col(rr, "distance_mean", "distance_xy_proxy")
        stable_frac = get_col(rr, "stable_frac", default=0.0)
        valid_frac = get_col(rr, "valid_frac", default=1.0)
        rollmax = abs(get_col(rr, "rollmax_mean", "proprio_roll_abs_max"))
        pitchmax = abs(get_col(rr, "pitchmax_mean", "proprio_pitch_abs_max"))

        old_rgated = get_col(rr, "R_gated_mean")
        old_motion = clamp(get_col(rr, "R_motion_mean"))
        old_stability = clamp(get_col(rr, "R_stability_mean"))
        old_energy = clamp(get_col(rr, "R_energy_mean"))
        old_aux = clamp(get_col(rr, "R_aux_mean"))

        expected_min_z = max(args.min_absolute_z, ref_h - args.allowed_body_drop)
        target_height_margin = zmin - expected_min_z

        # Smooth score:
        # margin >= +soft/2 => near 1
        # margin <= -soft/2 => near 0
        height_score = clamp((target_height_margin + 0.5 * args.height_soft_margin) / args.height_soft_margin)

        # Keep absolute below22 as a diagnostic, but do not let it dominate low-body strategies.
        below22_score = clamp(1.0 - below22)

        orientation_score = clamp(
            1.0
            - 0.50 * min(1.0, rollmax / 0.50)
            - 0.50 * min(1.0, pitchmax / 0.50)
        )

        stability_v2 = clamp(
            0.25 * old_stability
            + 0.30 * height_score
            + 0.20 * stable_frac
            + 0.10 * valid_frac
            + 0.10 * orientation_score
            + 0.05 * below22_score
        )

        progress_norm = clamp(dist / max_dist)
        speed_norm = clamp(max(0.0, ref_vx) / max_vx)

        # Motion-aware but still safety-compatible.
        motion_v2 = clamp(
            0.20 * old_motion
            + 0.50 * progress_norm
            + 0.30 * speed_norm
        )

        # Energy still uses existing proxy for now.
        energy_v2 = clamp(old_energy)

        raw_v2 = clamp(
            b_motion * motion_v2
            + b_stability * stability_v2
            + b_energy * energy_v2
        )

        # Penalize aux and strong target-height collapse.
        collapse_penalty = clamp(max(0.0, -target_height_margin) / args.height_soft_margin)
        gated_v2 = clamp(raw_v2 * math.exp(-args.aux_penalty * old_aux) * math.exp(-0.75 * collapse_penalty))

        rr["R_gated_old_mean"] = f"{old_rgated:.9f}"
        rr["expected_min_z_v2"] = f"{expected_min_z:.9f}"
        rr["target_height_margin_v2"] = f"{target_height_margin:.9f}"
        rr["R_height_target_v2_mean"] = f"{height_score:.9f}"
        rr["R_orientation_v2_mean"] = f"{orientation_score:.9f}"
        rr["R_stability_v2_mean"] = f"{stability_v2:.9f}"
        rr["R_motion_v2_mean"] = f"{motion_v2:.9f}"
        rr["R_energy_v2_mean"] = f"{energy_v2:.9f}"
        rr["R_progress_norm_v2_mean"] = f"{progress_norm:.9f}"
        rr["R_speed_norm_v2_mean"] = f"{speed_norm:.9f}"
        rr["R_profile_v2_raw_mean"] = f"{raw_v2:.9f}"
        rr["R_profile_v2_gated_mean"] = f"{gated_v2:.9f}"

        # Compatibility with existing candidate-ranker trainer.
        rr["R_gated_mean"] = f"{gated_v2:.9f}"

        out.append(rr)

    write_csv(Path(args.output_csv), out)

    print(f"[TRACER] input:  {args.input_csv}")
    print(f"[TRACER] output: {args.output_csv}")
    print(f"[TRACER] rows:   {len(out)}")
    print(f"[TRACER] max_dist={max_dist:.6f}")
    print()

    for profile in sorted(set(str(r.get("profile", "")) for r in out)):
        sub = [r for r in out if str(r.get("profile", "")) == profile]
        sub.sort(key=lambda x: to_float(x.get("R_profile_v2_gated_mean")), reverse=True)

        print("=" * 80)
        print(f"profile={profile}")
        for r in sub[:8]:
            print(
                f"  {r.get('preset',''):34s} "
                f"score_v2={to_float(r.get('R_profile_v2_gated_mean')):.4f} "
                f"old={to_float(r.get('R_gated_old_mean')):.4f} "
                f"stable={to_float(r.get('stable_frac')):.2f} "
                f"zmin={to_float(r.get('zmin_mean')):.3f} "
                f"expected_min_z={to_float(r.get('expected_min_z_v2')):.3f} "
                f"margin={to_float(r.get('target_height_margin_v2')):.3f} "
                f"height_score={to_float(r.get('R_height_target_v2_mean')):.3f} "
                f"dist={to_float(r.get('distance_mean')):.4f} "
                f"vx={to_float(r.get('ref_vx')):.3f} "
                f"h={to_float(r.get('ref_body_height')):.3f} "
                f"c={to_float(r.get('ref_swing_clearance')):.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
