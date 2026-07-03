#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


def risk_name_from_state(risk):
    mismatch = float(risk.get("ram_mismatch", 0.0))
    unc = float(risk.get("ram_uncertainty", 0.0))
    yaw = float(risk.get("recent_max_yaw_rate", 0.0))
    slip = float(risk.get("recent_slip_score", 0.0))
    stab = float(risk.get("body_stability_score", 1.0))
    sat = float(risk.get("yaw_saturation_fraction", 0.0))

    if stab < 0.78 or mismatch > 0.35 or unc > 0.35 or slip > 0.25 or yaw > 0.28 or sat > 0.20:
        return "high"

    if mismatch > 0.10 or unc > 0.10 or slip > 0.08 or yaw > 0.16:
        return "moderate"

    return "low"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-json", default="configs/phase_b_theta5_teacher_table_v0/trusted_table.json")
    ap.add_argument("--linear-model", required=True)
    ap.add_argument("--risk-state-json", required=True)
    ap.add_argument("--world-name", default="earth")
    ap.add_argument("--goal-distance-ahead", type=float, default=0.5)
    ap.add_argument("--goal-stop-distance", type=float, default=0.15)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    with open(args.risk_state_json, "r") as f:
        risk = json.load(f)

    risk_name = risk_name_from_state(risk)
    key = f"{args.world_name}::{risk_name}"

    table_hit = False
    table_entry = None

    if Path(args.table_json).is_file():
        with open(args.table_json, "r") as f:
            table = json.load(f)
        table_entry = table.get("best_table", {}).get(key)
        table_hit = table_entry is not None

    if table_hit:
        reached_rate = float(table_entry.get("reached_rate", 0.0))
        n_teacher = int(table_entry.get("n", 0))

        if reached_rate < 0.80:
            table_hit = False
        else:
            action = dict(table_entry["action"])

            profile = {
                "name": "theta5_teacher_table_v0",
                "vx_far": action["vx_far"],
                "vx_near": action["vx_near"],
                "goal_slow_distance": action["goal_slow_distance"],
                "goal_stop_distance": args.goal_stop_distance,
                "goal_distance_ahead": args.goal_distance_ahead,
                "body_height": action["body_height"],
                "swing_clearance": action["swing_clearance"],
                "world_name": args.world_name,
            }

            guard_reasons = [
                f"teacher-table hit: {key}",
                f"teacher_case={table_entry.get('case_name')}",
                f"teacher_reached_rate={table_entry.get('reached_rate')}",
                f"teacher_n={table_entry.get('n')}",
            ]

            if n_teacher < 3:
                guard_reasons.append("low teacher evidence: n<3")

            out = {
                "model": args.linear_model,
                "table_json": args.table_json,
                "source": "teacher_table",
                "table_key": key,
                "risk_name": risk_name,
                "risk_state_json": args.risk_state_json,
                "risk_state": risk,
                "world_name": args.world_name,
                "raw_prediction": action,
                "profile": profile,
                "guard_reasons": guard_reasons,
                "teacher_entry": table_entry,
            }

    if not table_hit:
        fallback_reason = f"table miss: {key}"
        if table_entry is not None:
            fallback_reason = f"untrusted teacher table entry: {key}, reached_rate={table_entry.get('reached_rate')}"

        cmd = [
            sys.executable,
            "scripts/runtime/tracer_predict_phase_b_theta5_profile_v0.py",
            "--model", args.linear_model,
            "--risk-state-json", args.risk_state_json,
            "--world-name", args.world_name,
            "--goal-distance-ahead", str(args.goal_distance_ahead),
            "--goal-stop-distance", str(args.goal_stop_distance),
        ]
        raw = subprocess.check_output(cmd, text=True)
        out = json.loads(raw)
        out["source"] = "linear_fallback"
        out["table_json"] = args.table_json
        out["table_key"] = key
        out["risk_name"] = risk_name
        out["guard_reasons"] = [fallback_reason] + out.get("guard_reasons", [])

    text = json.dumps(out, indent=2, sort_keys=True)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            f.write(text + "\n")

    print(text)


if __name__ == "__main__":
    main()
