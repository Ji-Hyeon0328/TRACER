#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-json", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    pred = json.load(open(args.prediction_json))
    profile = pred.get("profile", {})
    decision = pred.get("runtime_decision", {})

    export = {
        "schema": "tracer_phase_b_runtime_decision_v0",
        "world_name": profile.get("world_name") or pred.get("world_name"),
        "theta_profile": {
            "name": profile.get("name"),
            "vx_far": profile.get("vx_far"),
            "vx_near": profile.get("vx_near"),
            "goal_slow_distance": profile.get("goal_slow_distance"),
            "goal_stop_distance": profile.get("goal_stop_distance"),
            "body_height": profile.get("body_height"),
            "swing_clearance": profile.get("swing_clearance"),
        },
        "objective": {
            "semantic": decision.get("semantic"),
            "calibrated_semantic": decision.get("calibrated_semantic"),
            "beta_velocity": decision.get("beta_velocity"),
            "beta_stability": decision.get("beta_stability"),
            "beta_energy": decision.get("beta_energy"),
        },
        "ram": {
            "future_risk": decision.get("future_risk"),
            "future_uncertainty": decision.get("future_uncertainty"),
            "calibrated_future_risk": decision.get("calibrated_future_risk"),
            "future_invalid": decision.get("future_invalid"),
            "calibrated_future_invalid": decision.get("calibrated_future_invalid"),
            "recovery_needed": decision.get("recovery_needed"),
            "calibrated_recovery_needed": decision.get("calibrated_recovery_needed"),
            "high_variability": decision.get("high_variability"),
            "normal_walk_blocked": decision.get("normal_walk_blocked"),
            "uncertainty_n": decision.get("uncertainty_n"),
            "uncertainty_majority_semantic": decision.get("uncertainty_majority_semantic"),
        },
        "recommended_gate": {
            "allow_normal_walk": not bool(decision.get("normal_walk_blocked", False)),
            "requires_recovery_or_alternative_primitive": bool(decision.get("normal_walk_blocked", False)),
            "reason": decision.get("calibrated_semantic") or decision.get("semantic"),
        },
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(export, indent=2, sort_keys=True) + "\n")
    print(json.dumps(export, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
