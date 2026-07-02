#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def as_bool(x: Any) -> bool:
    return bool(x)


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None


def rate(xs):
    return sum(1 for x in xs if x) / len(xs) if xs else None


def load_rows(root: Path, terrain: str, policy_contains: str):
    rows = []

    for summary_path in sorted(root.glob("runtime_rollout_*/summaries/*.json")):
        try:
            s = json.loads(summary_path.read_text())
        except Exception:
            continue

        ep = str(s.get("episode_id", ""))
        pol = str(s.get("policy_id", ""))

        if terrain and str(s.get("terrain", "")) != terrain:
            continue
        if policy_contains and policy_contains not in ep and policy_contains not in pol:
            continue

        s["_summary_path"] = str(summary_path)
        rows.append(s)

    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/rollouts/phase_a_runtime_v0")
    ap.add_argument("--terrain", required=True)
    ap.add_argument("--policy-contains", default="")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)

    # Command-mode thresholds.
    ap.add_argument("--hold-vx-eps", type=float, default=0.005)

    # Stream freshness thresholds.
    ap.add_argument("--min-debug-fresh-rate", type=float, default=0.80)
    ap.add_argument("--max-proprio-age-p90", type=float, default=0.50)
    ap.add_argument("--max-odom-age-p90", type=float, default=0.50)

    # Hold stability thresholds.
    ap.add_argument("--min-hold-zmin", type=float, default=0.18)
    ap.add_argument("--max-hold-below22-frac", type=float, default=0.10)
    ap.add_argument("--max-hold-roll", type=float, default=0.25)
    ap.add_argument("--max-hold-pitch", type=float, default=0.25)

    # Locomotion progress threshold for explicit v1 metric.
    ap.add_argument("--min-locomotion-distance", type=float, default=0.08)

    args = ap.parse_args()

    root = Path(args.root)
    summaries = load_rows(root, args.terrain, args.policy_contains)

    print(f"[TRACER] root={root}")
    print(f"[TRACER] terrain={args.terrain}")
    print(f"[TRACER] policy_contains={args.policy_contains}")
    print(f"[TRACER] n={len(summaries)}")

    if not summaries:
        raise SystemExit("[TRACER][ERR] no matching runtime rollout summaries")

    out_rows = []

    for i, s in enumerate(summaries, start=1):
        episode_id = str(s.get("episode_id", ""))
        policy_id = str(s.get("policy_id", ""))

        vx = as_float(s.get("mpc_vx_mean"))
        body_h = as_float(s.get("mpc_body_height_mean"))
        clearance = as_float(s.get("mpc_clearance_mean"))

        success_proxy = as_bool(s.get("success_proxy"))
        valid_data = as_bool(s.get("valid_data"))
        height_stable = as_bool(s.get("proprio_base_height_stable"))

        distance = as_float(s.get("distance_xy_proxy"))
        zmin = as_float(s.get("proprio_base_z_min"))
        z50 = as_float(s.get("proprio_base_z_p50"))
        below18 = as_float(s.get("proprio_base_z_below_0p18_frac"))
        below22 = as_float(s.get("proprio_base_z_below_0p22_frac"))
        rollmax = as_float(s.get("proprio_roll_abs_max"))
        pitchmax = as_float(s.get("proprio_pitch_abs_max"))
        debug_fresh = as_float(s.get("debug_fresh_rate_0p5s"))
        age_prop = as_float(s.get("age_proprio_p90"))
        age_odom = as_float(s.get("age_odom_p90"))

        command_mode = "hold_or_recovery" if abs(vx) <= args.hold_vx_eps else "locomotion"

        stream_fresh_proxy = (
            debug_fresh >= args.min_debug_fresh_rate
            and age_prop <= args.max_proprio_age_p90
            and age_odom <= args.max_odom_age_p90
        )

        hold_stable_proxy = (
            command_mode == "hold_or_recovery"
            and stream_fresh_proxy
            and height_stable
            and zmin >= args.min_hold_zmin
            and below22 <= args.max_hold_below22_frac
            and rollmax <= args.max_hold_roll
            and pitchmax <= args.max_hold_pitch
        )

        locomotion_success_proxy_v1 = (
            command_mode == "locomotion"
            and stream_fresh_proxy
            and height_stable
            and distance >= args.min_locomotion_distance
            and below22 <= args.max_hold_below22_frac
        )

        low_height_warning = below22 > 0.0
        drift_warning_hold = command_mode == "hold_or_recovery" and distance > args.min_locomotion_distance

        row = {
            "idx": i,
            "episode_id": episode_id,
            "policy_id": policy_id,
            "terrain": s.get("terrain"),
            "summary_path": s.get("_summary_path"),
            "command_mode": command_mode,
            "mpc_vx_mean": vx,
            "mpc_body_height_mean": body_h,
            "mpc_clearance_mean": clearance,
            "success_proxy_raw": success_proxy,
            "valid_data_raw": valid_data,
            "stream_fresh_proxy": stream_fresh_proxy,
            "height_stable_proxy": height_stable,
            "hold_stable_proxy": hold_stable_proxy,
            "locomotion_success_proxy_v1": locomotion_success_proxy_v1,
            "low_height_warning": low_height_warning,
            "drift_warning_hold": drift_warning_hold,
            "distance_xy_proxy": distance,
            "proprio_base_z_min": zmin,
            "proprio_base_z_p50": z50,
            "proprio_base_z_below_0p18_frac": below18,
            "proprio_base_z_below_0p22_frac": below22,
            "proprio_roll_abs_max": rollmax,
            "proprio_pitch_abs_max": pitchmax,
            "debug_fresh_rate_0p5s": debug_fresh,
            "age_proprio_p90": age_prop,
            "age_odom_p90": age_odom,
        }

        out_rows.append(row)

        print()
        print(f"---- row {i} {episode_id}")
        print(f"command_mode: {command_mode}")
        print(f"vx/h/clr: {vx:.4f}, {body_h:.4f}, {clearance:.4f}")
        print(f"raw success_proxy: {success_proxy}")
        print(f"raw valid_data: {valid_data}")
        print(f"stream_fresh_proxy: {stream_fresh_proxy}")
        print(f"height_stable_proxy: {height_stable}")
        print(f"hold_stable_proxy: {hold_stable_proxy}")
        print(f"locomotion_success_proxy_v1: {locomotion_success_proxy_v1}")
        print(f"distance_xy_proxy: {distance}")
        print(f"zmin/z50/below22: {zmin}, {z50}, {below22}")
        print(f"roll/pitch max: {rollmax}, {pitchmax}")

    hold_rows = [r for r in out_rows if r["command_mode"] == "hold_or_recovery"]
    loco_rows = [r for r in out_rows if r["command_mode"] == "locomotion"]

    agg = {
        "version": "phase_a_runtime_aggregate_v1_modes",
        "n": len(out_rows),
        "terrain": args.terrain,
        "policy_contains": args.policy_contains,
        "raw_success_rate": rate([r["success_proxy_raw"] for r in out_rows]),
        "raw_valid_rate": rate([r["valid_data_raw"] for r in out_rows]),
        "stream_fresh_rate": rate([r["stream_fresh_proxy"] for r in out_rows]),
        "height_stable_rate": rate([r["height_stable_proxy"] for r in out_rows]),
        "hold_command_rate": len(hold_rows) / len(out_rows),
        "locomotion_command_rate": len(loco_rows) / len(out_rows),
        "hold_stable_rate_all": rate([r["hold_stable_proxy"] for r in out_rows]),
        "hold_stable_rate_among_hold": rate([r["hold_stable_proxy"] for r in hold_rows]),
        "locomotion_success_rate_v1_all": rate([r["locomotion_success_proxy_v1"] for r in out_rows]),
        "locomotion_success_rate_v1_among_locomotion": rate([r["locomotion_success_proxy_v1"] for r in loco_rows]),
        "low_height_warning_rate": rate([r["low_height_warning"] for r in out_rows]),
        "hold_drift_warning_rate_among_hold": rate([r["drift_warning_hold"] for r in hold_rows]),
        "distance_mean": mean([r["distance_xy_proxy"] for r in out_rows]),
        "hold_distance_mean": mean([r["distance_xy_proxy"] for r in hold_rows]),
        "locomotion_distance_mean": mean([r["distance_xy_proxy"] for r in loco_rows]),
        "zmin_mean": mean([r["proprio_base_z_min"] for r in out_rows]),
        "z50_mean": mean([r["proprio_base_z_p50"] for r in out_rows]),
        "below18_mean": mean([r["proprio_base_z_below_0p18_frac"] for r in out_rows]),
        "below22_mean": mean([r["proprio_base_z_below_0p22_frac"] for r in out_rows]),
        "rollmax_mean": mean([r["proprio_roll_abs_max"] for r in out_rows]),
        "pitchmax_mean": mean([r["proprio_pitch_abs_max"] for r in out_rows]),
        "debug_fresh_rate_0p5s_mean": mean([r["debug_fresh_rate_0p5s"] for r in out_rows]),
        "age_proprio_p90_mean": mean([r["age_proprio_p90"] for r in out_rows]),
        "age_odom_p90_mean": mean([r["age_odom_p90"] for r in out_rows]),
        "thresholds": {
            "hold_vx_eps": args.hold_vx_eps,
            "min_debug_fresh_rate": args.min_debug_fresh_rate,
            "max_proprio_age_p90": args.max_proprio_age_p90,
            "max_odom_age_p90": args.max_odom_age_p90,
            "min_hold_zmin": args.min_hold_zmin,
            "max_hold_below22_frac": args.max_hold_below22_frac,
            "max_hold_roll": args.max_hold_roll,
            "max_hold_pitch": args.max_hold_pitch,
            "min_locomotion_distance": args.min_locomotion_distance,
        },
    }

    print()
    print("==== aggregate v1 modes ====")
    print(json.dumps(agg, indent=2, sort_keys=True))

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps({"aggregate": agg, "rows": out_rows}, indent=2, sort_keys=True) + "\n")

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"[TRACER] wrote json: {out_json}")
    print(f"[TRACER] wrote csv: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
