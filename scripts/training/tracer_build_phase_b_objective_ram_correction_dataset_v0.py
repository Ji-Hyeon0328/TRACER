#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path


def ff(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def bb(x):
    return str(x).lower() == "true"


def beta_from_actual(row):
    actual_sem = row["actual_semantic"]
    yaw_sat = bb(row["actual_yaw_saturated"])

    if actual_sem == "stable_goal_reach_flat_locomotion":
        beta = {"beta_velocity": 0.45, "beta_stability": 0.40, "beta_energy": 0.15}
        recovery = False
    elif actual_sem == "approach_possible_but_post_reach_hold_needed":
        beta = {"beta_velocity": 0.08, "beta_stability": 0.82, "beta_energy": 0.10}
        recovery = True
    elif actual_sem == "cautious_probe_required":
        beta = {"beta_velocity": 0.12, "beta_stability": 0.76, "beta_energy": 0.12}
        recovery = True
    else:
        beta = {"beta_velocity": 0.05, "beta_stability": 0.85, "beta_energy": 0.10}
        recovery = True

    if yaw_sat:
        beta["beta_stability"] = min(0.90, beta["beta_stability"] + 0.05)
        beta["beta_velocity"] = max(0.03, beta["beta_velocity"] - 0.03)

    z = sum(beta.values())
    beta = {k: v / z for k, v in beta.items()}
    return beta, recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-csv", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.score_csv)))
    examples = []

    for r in rows:
        # Use only integrated runs that actually had predictions.
        if not r.get("pred_semantic") or r.get("pred_semantic") == "None":
            continue

        beta, recovery_needed = beta_from_actual(r)

        action = {
            "vx_far": ff(r.get("vx_far")),
            "vx_near": ff(r.get("vx_near")),
            "goal_slow_distance": 0.0,  # may be missing from summary; predictor can still use available theta dims later
            "body_height": ff(r.get("body_height")),
            "swing_clearance": ff(r.get("swing_clearance")),
        }

        # Recover slow distance from profile name is not reliable, so leave zero unless summary contains it.
        # If future summaries include goal_slow_distance, this will be populated.
        if r.get("goal_slow_distance"):
            action["goal_slow_distance"] = ff(r.get("goal_slow_distance"))

        ex = {
            "dataset_type": "phase_b_objective_ram_correction_v0",
            "source_score_csv": args.score_csv,
            "source_run_dir": r.get("run_dir"),
            "world_name": r.get("world_name"),
            "case_name": r.get("theta5_profile_name"),
            "obs": {
                "terrain": {
                    "world_name": r.get("world_name"),
                    "terrain_family": "sponge" if "sponge" in str(r.get("world_name")) else "flat",
                    "soft_contact": "sponge" in str(r.get("world_name")),
                },
                "candidate_action": action,
            },
            "objective_target": {
                **beta,
                "semantic": r["actual_semantic"],
                "recovery_needed": recovery_needed,
            },
            "ram_target": {
                "future_risk": ff(r.get("actual_future_risk")),
                "future_invalid": bb(r.get("actual_future_invalid")),
                "recovery_needed": bb(r.get("actual_recovery_needed")),
                "approach_success": bb(r.get("actual_approach_success")),
                "stable_reached": bb(r.get("actual_stable_reached")),
                "drift_after_approach": bb(r.get("actual_drift_after_approach")),
                "yaw_saturated": bb(r.get("actual_yaw_saturated")),
                "final_rel_dist": ff(r.get("actual_final_rel_dist")),
                "min_rel_dist": ff(r.get("actual_min_rel_dist")),
                "abs_odom_x_delta": ff(r.get("actual_abs_odom_x_delta")),
                "max_abs_yaw_rate": ff(r.get("actual_max_abs_yaw_rate")),
            },
            "prediction_error": {
                "pred_semantic": r.get("pred_semantic"),
                "actual_semantic": r.get("actual_semantic"),
                "semantic_match": bb(r.get("semantic_match")),
                "risk_error": ff(r.get("risk_error")),
                "approach_match": bb(r.get("approach_match")),
                "stable_match": bb(r.get("stable_match")),
                "recovery_match": bb(r.get("recovery_match")),
            },
        }

        examples.append(ex)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_jsonl, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, sort_keys=True) + "\n")

    semantic_counts = {}
    for ex in examples:
        sem = ex["objective_target"]["semantic"]
        semantic_counts[sem] = semantic_counts.get(sem, 0) + 1

    summary = {
        "num_examples": len(examples),
        "semantic_counts": semantic_counts,
        "source_score_csv": args.score_csv,
        "out_jsonl": args.out_jsonl,
    }

    with open(args.out_summary_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
