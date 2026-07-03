#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


def latest_model(pattern, filename):
    roots = sorted(Path("artifacts").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in roots:
        f = r / filename
        if f.is_file():
            return str(f)
    return ""


def run_json(cmd):
    raw = subprocess.check_output(cmd, text=True)
    return json.loads(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta5-model", default="")
    ap.add_argument("--objective-ram-model", default="")
    ap.add_argument("--risk-state-json", required=True)
    ap.add_argument("--world-name", default="earth")
    ap.add_argument("--goal-distance-ahead", type=float, default=0.5)
    ap.add_argument("--goal-stop-distance", type=float, default=0.15)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    theta5_model = args.theta5_model or latest_model(
        "phase_b_theta5_regressor_v0_*",
        "phase_b_theta5_regressor_v0.json",
    )
    obj_ram_model = args.objective_ram_model
    if not obj_ram_model:
        cfg_model = Path("configs/phase_b_objective_ram_bootstrap_v0/current_model.json")
        if cfg_model.is_file():
            obj_ram_model = str(cfg_model)
        else:
            obj_ram_model = latest_model(
                "phase_b_objective_ram_bootstrap_v0_*",
                "phase_b_objective_ram_bootstrap_v0.json",
            )

    if not theta5_model:
        raise RuntimeError("theta5 model not found")
    if not obj_ram_model:
        raise RuntimeError("objective/RAM bootstrap model not found")

    theta5 = run_json([
        sys.executable,
        "scripts/runtime/tracer_predict_phase_b_theta5_hybrid_profile_v0.py",
        "--linear-model", theta5_model,
        "--risk-state-json", args.risk_state_json,
        "--world-name", args.world_name,
        "--goal-distance-ahead", str(args.goal_distance_ahead),
        "--goal-stop-distance", str(args.goal_stop_distance),
    ])

    p = theta5["profile"]

    obj_ram = run_json([
        sys.executable,
        "scripts/runtime/tracer_predict_phase_b_objective_ram_bootstrap_v0.py",
        "--model", obj_ram_model,
        "--world-name", args.world_name,
        "--vx-far", str(p["vx_far"]),
        "--vx-near", str(p["vx_near"]),
        "--goal-slow-distance", str(p["goal_slow_distance"]),
        "--body-height", str(p["body_height"]),
        "--swing-clearance", str(p["swing_clearance"]),
    ])

    objective = obj_ram["objective_prediction"]
    ram = obj_ram["ram_prediction"]

    out = {
        # Keep top-level compatibility with theta5 runner.
        "model": theta5_model,
        "model_type": theta5.get("model_type"),
        "source": theta5.get("source"),
        "table_key": theta5.get("table_key"),
        "risk_name": theta5.get("risk_name"),
        "risk_state_json": args.risk_state_json,
        "risk_state": theta5.get("risk_state"),
        "world_name": args.world_name,
        "raw_prediction": theta5.get("raw_prediction", {}),
        "profile": p,
        "guard_reasons": theta5.get("guard_reasons", []),

        # New Objective/RAM layer.
        "theta5_prediction": theta5,
        "objective_ram_prediction": obj_ram,
        "runtime_decision": {
            "semantic": objective.get("semantic"),
            "beta_velocity": objective.get("beta_velocity"),
            "beta_stability": objective.get("beta_stability"),
            "beta_energy": objective.get("beta_energy"),
            "future_risk": ram.get("future_risk"),
            "future_invalid": ram.get("future_invalid"),
            "recovery_needed": ram.get("recovery_needed"),
            "stable_reached_pred": ram.get("stable_reached"),
            "approach_success_pred": ram.get("approach_success"),
            "drift_after_approach_pred": ram.get("drift_after_approach"),
            "yaw_saturated_pred": ram.get("yaw_saturated"),
        },
    }

    text = json.dumps(out, indent=2, sort_keys=True)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            f.write(text + "\n")

    print(text)


if __name__ == "__main__":
    main()
