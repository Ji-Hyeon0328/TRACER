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
        raise RuntimeError("No rows to write.")

    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-csv",
        default="data/phase_a_reference_banks_earth_v0/phase_a_reference_candidates_merged_v0.csv",
    )
    ap.add_argument(
        "--output-csv",
        default="data/phase_a_reference_banks_earth_v1/phase_a_reference_candidates_motionaware_v1.csv",
    )
    ap.add_argument("--max-command-vx", type=float, default=0.12)
    ap.add_argument("--aux-penalty", type=float, default=1.25)
    args = ap.parse_args()

    rows = read_csv(Path(args.input_csv))
    if not rows:
        raise RuntimeError(f"No rows found: {args.input_csv}")

    max_dist = max(1e-6, max(to_float(r.get("distance_mean")) for r in rows))
    max_vx = max(1e-6, args.max_command_vx)

    out_rows: list[dict[str, Any]] = []

    for r in rows:
        rr = dict(r)

        beta = parse_beta(rr.get("beta", ""))
        b_m, b_s, b_e = beta[0], beta[1], beta[2]

        old_rgated = to_float(rr.get("R_gated_mean"))
        old_motion = clamp(to_float(rr.get("R_motion_mean")))
        stability = clamp(to_float(rr.get("R_stability_mean")))
        energy = clamp(to_float(rr.get("R_energy_mean")))
        aux = clamp(to_float(rr.get("R_aux_mean")))

        vx = max(0.0, to_float(rr.get("ref_vx")))
        dist = max(0.0, to_float(rr.get("distance_mean")))

        progress_norm = clamp(dist / max_dist)
        speed_norm = clamp(vx / max_vx)

        # Motion-aware score:
        # old R_motion alone was too weak and allowed low-speed candidates to dominate
        # even for motion_extreme. This version explicitly rewards actual progress
        # and commanded forward speed while retaining the old tracking/stability proxy.
        motion_v1 = clamp(
            0.20 * old_motion
            + 0.50 * progress_norm
            + 0.30 * speed_norm
        )

        raw_v1 = clamp(
            b_m * motion_v1
            + b_s * stability
            + b_e * energy
        )

        gated_v1 = clamp(raw_v1 * math.exp(-args.aux_penalty * aux))

        rr["R_gated_old_mean"] = f"{old_rgated:.9f}"
        rr["R_motion_v1_mean"] = f"{motion_v1:.9f}"
        rr["R_progress_norm_mean"] = f"{progress_norm:.9f}"
        rr["R_speed_norm_mean"] = f"{speed_norm:.9f}"
        rr["R_profile_v1_raw_mean"] = f"{raw_v1:.9f}"
        rr["R_profile_v1_gated_mean"] = f"{gated_v1:.9f}"

        # Compatibility: existing candidate ranker trainer reads R_gated_mean.
        # For this v1 CSV, R_gated_mean intentionally means motion-aware profile score.
        rr["R_gated_mean"] = f"{gated_v1:.9f}"

        out_rows.append(rr)

    write_csv(Path(args.output_csv), out_rows)

    print(f"[TRACER] input:  {args.input_csv}")
    print(f"[TRACER] output: {args.output_csv}")
    print(f"[TRACER] rows:   {len(out_rows)}")
    print(f"[TRACER] max_dist={max_dist:.6f} max_command_vx={max_vx:.6f}")
    print()
    print("===== top by profile, motion-aware v1 =====")

    profiles = sorted(set(str(r.get("profile", "")) for r in out_rows))
    for p in profiles:
        subset = [r for r in out_rows if str(r.get("profile", "")) == p]
        subset.sort(key=lambda r: to_float(r.get("R_profile_v1_gated_mean")), reverse=True)
        print()
        print(f"profile={p}")
        for r in subset[:6]:
            print(
                f"  {r.get('preset',''):32s} "
                f"score_v1={to_float(r.get('R_profile_v1_gated_mean')):.4f} "
                f"old={to_float(r.get('R_gated_old_mean')):.4f} "
                f"motion_v1={to_float(r.get('R_motion_v1_mean')):.4f} "
                f"progress={to_float(r.get('R_progress_norm_mean')):.3f} "
                f"speed={to_float(r.get('R_speed_norm_mean')):.3f} "
                f"vx={to_float(r.get('ref_vx')):.3f} "
                f"h={to_float(r.get('ref_body_height')):.3f} "
                f"c={to_float(r.get('ref_swing_clearance')):.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
