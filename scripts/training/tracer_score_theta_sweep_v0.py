#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(key, default))
        return x if math.isfinite(x) else default
    except Exception:
        return default


def b(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).lower() == "true"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))



def command_effort(row: dict) -> float:
    """Command-based effort proxy.

    Prefer explicit command_effort_proxy from runtime summaries.
    Fallback to mpc_reference if available.
    """
    if "command_effort_proxy" in row and str(row.get("command_effort_proxy", "")).strip() != "":
        try:
            return max(0.0, float(row["command_effort_proxy"]))
        except Exception:
            pass

    try:
        import json
        import ast

        ref_s = str(row.get("mpc_reference", "")).strip()
        if not ref_s:
            return 0.0

        ref = json.loads(ref_s) if ref_s.startswith("[") else ast.literal_eval(ref_s)

        vx = abs(float(ref[1])) if len(ref) > 1 else 0.0
        yaw = abs(float(ref[2])) if len(ref) > 2 else 0.0
        h = float(ref[3]) if len(ref) > 3 else 0.30
        clr = abs(float(ref[4])) if len(ref) > 4 else 0.035

        return 0.45 * vx + 0.10 * yaw + 0.20 * abs(h - 0.30) + 0.25 * clr
    except Exception:
        return 0.0

def score_row(row: dict[str, Any]) -> dict[str, Any]:
    beta_m = f(row, "beta_motion", 1.0 / 3.0)
    beta_s = f(row, "beta_stability", 1.0 / 3.0)
    beta_e = f(row, "beta_energy", 1.0 / 3.0)
    beta_sum = max(1e-9, beta_m + beta_s + beta_e)
    beta_m, beta_s, beta_e = beta_m / beta_sum, beta_s / beta_sum, beta_e / beta_sum

    distance = f(row, "distance_xy_proxy", 0.0)
    z_mean = f(row, "proprio_base_z_mean", 0.0)
    z_min = f(row, "proprio_base_z_min", 0.0)
    roll = f(row, "proprio_roll_abs_max", 999.0)
    pitch = f(row, "proprio_pitch_abs_max", 999.0)
    vx = abs(f(row, "ref_vx", f(row, "mpc_vx_mean", 0.0)))
    body_h = f(row, "ref_body_height", f(row, "mpc_body_height_mean", 0.30))
    clearance = f(row, "ref_swing_clearance", f(row, "mpc_clearance_mean", 0.035))
    stable = b(row, "proprio_base_height_stable")

    # Motion: distance/progress proxy. 0.08m in 3 sec is treated as strong for current setup.
    r_motion = clamp01(distance / 0.08)

    # Stability: height + attitude. Penalize below 0.22 strongly.
    z_min_score = clamp01((z_min - 0.18) / 0.08)
    z_mean_score = clamp01((z_mean - 0.20) / 0.10)
    roll_score = clamp01(1.0 - roll / 0.25)
    pitch_score = clamp01(1.0 - pitch / 0.35)
    r_stability = 0.30 * z_min_score + 0.25 * z_mean_score + 0.20 * roll_score + 0.25 * pitch_score
    if not stable:
        r_stability *= 0.35

    # Energy proxy: lower command effort is better, but do not over-reward zero-progress.
    effort = (
        2.0 * vx
        + 3.0 * max(0.0, body_h - 0.30)
        + 3.5 * max(0.0, clearance - 0.035)
    )
    r_energy = math.exp(-2.0 * command_effort(row))

    r_beta = beta_m * r_motion + beta_s * r_stability + beta_e * r_energy

    out = dict(row)
    out.update({
        "R_motion_theta_v0": r_motion,
        "R_stability_theta_v0": r_stability,
        "R_energy_theta_v0": r_energy,
        "R_beta_theta_v0": r_beta,
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    inp = Path(args.summary_csv)
    out = Path(args.out_csv) if args.out_csv else inp.with_name("theta_sweep_scored_v0.csv")

    rows = list(csv.DictReader(inp.open()))
    scored = [score_row(r) for r in rows]
    scored.sort(key=lambda r: float(r["R_beta_theta_v0"]), reverse=True)

    fields = []
    for r in scored:
        for k in r:
            if k not in fields:
                fields.append(k)

    with out.open("w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fields)
        w.writeheader()
        w.writerows(scored)

    print("[TRACER] wrote", out)
    print()
    print("rank,preset,R_beta,R_motion,R_stability,R_energy,stable,distance,z_min,pitch")
    for i, r in enumerate(scored, 1):
        print(
            f"{i},{r.get('preset')},"
            f"{float(r['R_beta_theta_v0']):.4f},"
            f"{float(r['R_motion_theta_v0']):.4f},"
            f"{float(r['R_stability_theta_v0']):.4f},"
            f"{float(r['R_energy_theta_v0']):.4f},"
            f"{r.get('proprio_base_height_stable')},"
            f"{r.get('distance_xy_proxy')},"
            f"{r.get('proprio_base_z_min')},"
            f"{r.get('proprio_pitch_abs_max')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
