#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
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


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if math.isnan(x) or math.isinf(x):
        return lo
    return max(lo, min(hi, x))


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, "")
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def b(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"true", "1", "yes"}


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def std(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) >= 2 else 0.0


def score_episode(row: dict[str, Any]) -> dict[str, Any]:
    valid = b(row, "valid_data")
    stable = b(row, "proprio_base_height_stable")

    duration = f(row, "duration_sec", 5.0)
    vx = f(row, "ref_vx", f(row, "mpc_vx_mean", 0.0))
    yaw_rate = f(row, "ref_yaw_rate", 0.0)
    body_h = f(row, "ref_body_height", f(row, "mpc_body_height_mean", 0.30))
    clearance = f(row, "ref_swing_clearance", f(row, "mpc_clearance_mean", 0.035))

    dist = f(row, "distance_xy_proxy", 0.0)
    zmin = f(row, "proprio_base_z_min", 0.0)
    zmean = f(row, "proprio_base_z_mean", 0.0)
    below18 = f(row, "proprio_base_z_below_0p18_frac", 1.0)
    below22 = f(row, "proprio_base_z_below_0p22_frac", 1.0)
    rollmax = f(row, "proprio_roll_abs_max", 9.0)
    pitchmax = f(row, "proprio_pitch_abs_max", 9.0)
    yaw_delta = abs(f(row, "proprio_yaw_delta", 0.0))

    # Phase-A command-conditioned local target.
    # This is not yet global point-goal navigation. It says:
    # "track vx for duration_sec while keeping yaw/lateral behavior reasonable."
    target_distance = max(0.02, abs(vx) * max(0.0, duration))
    progress_ratio = clamp(dist / target_distance, 0.0, 1.0)
    target_match = clamp(1.0 - abs(dist - target_distance) / max(target_distance, 1e-6), 0.0, 1.0)
    yaw_score = clamp(1.0 - yaw_delta / 0.75, 0.0, 1.0)

    # Motion should prefer meaningful progress, but not reward uncontrolled flying/falling too much.
    R_motion = (
        0.50 * target_match
        + 0.35 * progress_ratio
        + 0.15 * yaw_score
    )

    zmin_score = clamp((zmin - 0.10) / (0.22 - 0.10), 0.0, 1.0)
    zmean_score = clamp((zmean - 0.16) / (0.26 - 0.16), 0.0, 1.0)
    below22_score = clamp(1.0 - below22, 0.0, 1.0)
    below18_score = clamp(1.0 - below18, 0.0, 1.0)
    roll_score = clamp(1.0 - rollmax / 0.80, 0.0, 1.0)
    pitch_score = clamp(1.0 - pitchmax / 0.55, 0.0, 1.0)
    stable_score = 1.0 if stable else 0.0

    R_stability = (
        0.20 * zmin_score
        + 0.20 * zmean_score
        + 0.20 * below22_score
        + 0.10 * below18_score
        + 0.15 * roll_score
        + 0.10 * pitch_score
        + 0.05 * stable_score
    )

    # Temporary energy proxy until torque/current logging is wired.
    # Penalize aggressive command magnitude and high posture/clearance.
    vx_eff = clamp(abs(vx) / 0.12, 0.0, 1.5)
    clearance_eff = clamp(max(0.0, clearance - 0.02) / 0.08, 0.0, 1.5)
    body_eff = clamp(abs(body_h - 0.30) / 0.06, 0.0, 1.5)
    R_energy = clamp(1.0 - (0.50 * vx_eff + 0.25 * clearance_eff + 0.25 * body_eff), 0.0, 1.0)

    invalid_pen = 0.35 if not valid else 0.0
    unstable_pen = 0.20 if not stable else 0.0
    low_pen = 0.25 * below22 + 0.20 * below18
    orient_pen = 0.10 * (1.0 - roll_score) + 0.10 * (1.0 - pitch_score)
    R_aux = clamp(invalid_pen + unstable_pen + low_pen + orient_pen, 0.0, 1.0)

    gate_valid = 1.0 if valid else 0.0
    gate_stability = clamp(1.0 - 0.75 * below22 - 0.50 * below18, 0.0, 1.0)
    if stable:
        gate_stability = max(gate_stability, 0.65)

    label = "risky"
    if not valid:
        label = "invalid"
    elif zmin < 0.09 or rollmax > 1.20 or pitchmax > 0.90:
        label = "failed"
    elif stable and zmin >= 0.17 and below22 <= 0.45 and rollmax <= 0.70 and pitchmax <= 0.55:
        label = "good"

    out = dict(row)
    out.update({
        "task_type": "forward_velocity_tracking_phase_a",
        "cmd_vx": vx,
        "cmd_yaw_rate": yaw_rate,
        "target_distance_x_proxy": target_distance,
        "target_yaw_delta": 0.0,
        "target_stop_vx": 0.0,
        "target_stop_yaw_rate": 0.0,
        "R_motion": R_motion,
        "R_stability": R_stability,
        "R_energy": R_energy,
        "R_aux": R_aux,
        "gate_valid": gate_valid,
        "gate_stability": gate_stability,
        "episode_label": label,
    })

    for profile, beta in PROFILE_BETA.items():
        raw = beta[0] * R_motion + beta[1] * R_stability + beta[2] * R_energy
        shaped = raw * math.exp(-0.75 * R_aux)
        gated = shaped * gate_valid * gate_stability
        out[f"R_raw_{profile}"] = raw
        out[f"R_shaped_{profile}"] = shaped
        out[f"R_gated_{profile}"] = gated

    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        wr = csv.DictWriter(fp, fieldnames=keys)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--min-valid-frac", type=float, default=1.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    summary_csv = run_dir / "reference_sweep_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)

    rows = list(csv.DictReader(summary_csv.open()))
    scored = [score_episode(r) for r in rows]

    scored_csv = run_dir / "reference_sweep_scored_phase_a_v0.csv"
    write_csv(scored_csv, scored)

    scored_jsonl = run_dir / "reference_sweep_scored_phase_a_v0.jsonl"
    with scored_jsonl.open("w", encoding="utf-8") as fp:
        for r in scored:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in scored:
        groups.setdefault((str(r["terrain"]), str(r["preset"])), []).append(r)

    profile_rows: list[dict[str, Any]] = []
    teacher_items: list[dict[str, Any]] = []

    for profile, beta in PROFILE_BETA.items():
        aggs: list[dict[str, Any]] = []

        for (terrain, preset), rs in groups.items():
            n = len(rs)
            valid_count = sum(str(r.get("valid_data", "")).lower() == "true" for r in rs)
            stable_count = sum(str(r.get("proprio_base_height_stable", "")).lower() == "true" for r in rs)
            good_count = sum(r.get("episode_label") == "good" for r in rs)
            failed_count = sum(r.get("episode_label") in {"failed", "invalid"} for r in rs)

            def vals(key: str) -> list[float]:
                return [f(r, key, float("nan")) for r in rs if not math.isnan(f(r, key, float("nan")))]

            first = rs[0]
            agg = {
                "profile": profile,
                "beta": json.dumps(beta),
                "terrain": terrain,
                "preset": preset,
                "n": n,
                "valid_count": valid_count,
                "valid_frac": valid_count / max(1, n),
                "stable_count": stable_count,
                "stable_frac": stable_count / max(1, n),
                "good_count": good_count,
                "failed_count": failed_count,
                "ref_vx": f(first, "ref_vx"),
                "ref_yaw_rate": f(first, "ref_yaw_rate"),
                "ref_body_height": f(first, "ref_body_height"),
                "ref_swing_clearance": f(first, "ref_swing_clearance"),
                "R_gated_mean": mean(vals(f"R_gated_{profile}")),
                "R_gated_std": std(vals(f"R_gated_{profile}")),
                "R_raw_mean": mean(vals(f"R_raw_{profile}")),
                "R_motion_mean": mean(vals("R_motion")),
                "R_stability_mean": mean(vals("R_stability")),
                "R_energy_mean": mean(vals("R_energy")),
                "R_aux_mean": mean(vals("R_aux")),
                "distance_mean": mean(vals("distance_xy_proxy")),
                "zmin_mean": mean(vals("proprio_base_z_min")),
                "zmean_mean": mean(vals("proprio_base_z_mean")),
                "below22_mean": mean(vals("proprio_base_z_below_0p22_frac")),
                "rollmax_mean": mean(vals("proprio_roll_abs_max")),
                "pitchmax_mean": mean(vals("proprio_pitch_abs_max")),
                "source_run_dir": str(run_dir),
            }
            aggs.append(agg)

        aggs.sort(
            key=lambda r: (
                float(r["R_gated_mean"]),
                float(r["stable_frac"]),
                float(r["R_stability_mean"]),
                float(r["R_motion_mean"]),
            ),
            reverse=True,
        )

        for rank, r in enumerate(aggs, start=1):
            r["rank"] = rank
            profile_rows.append(r)

        picked = 0
        for r in aggs:
            if float(r["valid_frac"]) < args.min_valid_frac:
                continue
            item = {
                "profile": profile,
                "beta": beta,
                "rank": int(r["rank"]),
                "terrain": r["terrain"],
                "preset": r["preset"],
                "action_phase_a": {
                    "vx": float(r["ref_vx"]),
                    "yaw_rate": float(r["ref_yaw_rate"]),
                    "body_height": float(r["ref_body_height"]),
                    "swing_clearance": float(r["ref_swing_clearance"]),
                    "enable": 1.0,
                },
                "metrics": {
                    "n": int(r["n"]),
                    "valid_frac": float(r["valid_frac"]),
                    "stable_frac": float(r["stable_frac"]),
                    "good_count": int(r["good_count"]),
                    "failed_count": int(r["failed_count"]),
                    "R_gated_mean": float(r["R_gated_mean"]),
                    "R_raw_mean": float(r["R_raw_mean"]),
                    "R_motion_mean": float(r["R_motion_mean"]),
                    "R_stability_mean": float(r["R_stability_mean"]),
                    "R_energy_mean": float(r["R_energy_mean"]),
                    "R_aux_mean": float(r["R_aux_mean"]),
                    "distance_mean": float(r["distance_mean"]),
                    "zmin_mean": float(r["zmin_mean"]),
                    "zmean_mean": float(r["zmean_mean"]),
                    "below22_mean": float(r["below22_mean"]),
                    "rollmax_mean": float(r["rollmax_mean"]),
                    "pitchmax_mean": float(r["pitchmax_mean"]),
                },
                "task_type": "forward_velocity_tracking_phase_a",
                "source_run_dir": str(run_dir),
                "note": (
                    "Phase-A teacher-bank sample from direct reference sweep. "
                    "This is for BC/offline warm-start of high-level meta-action, "
                    "not final full TRACER theta policy."
                ),
            }
            teacher_items.append(item)
            picked += 1
            if picked >= args.top_k:
                break

    profile_csv = run_dir / "reference_sweep_profile_rescore_phase_a_v0.csv"
    write_csv(profile_csv, profile_rows)

    teacher_jsonl = run_dir / "phase_a_teacher_bank_v0.jsonl"
    with teacher_jsonl.open("w", encoding="utf-8") as fp:
        for item in teacher_items:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[TRACER] wrote scored episodes: {scored_csv}")
    print(f"[TRACER] wrote scored jsonl:    {scored_jsonl}")
    print(f"[TRACER] wrote profile rescore: {profile_csv}")
    print(f"[TRACER] wrote teacher bank:    {teacher_jsonl}")
    print()

    for profile in PROFILE_BETA:
        rows_p = [r for r in profile_rows if r["profile"] == profile]
        print("=" * 80)
        print(f"profile={profile} beta={PROFILE_BETA[profile]}")
        for r in rows_p[: args.top_k]:
            print(
                f"{int(r['rank']):2d}. {r['preset']:32s} "
                f"Rgated={float(r['R_gated_mean']):.4f} "
                f"stable={int(r['stable_count'])}/{int(r['n'])} "
                f"dist={float(r['distance_mean']):.4f} "
                f"zmin={float(r['zmin_mean']):.4f} "
                f"below22={float(r['below22_mean']):.3f} "
                f"roll={float(r['rollmax_mean']):.3f} "
                f"pitch={float(r['pitchmax_mean']):.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
