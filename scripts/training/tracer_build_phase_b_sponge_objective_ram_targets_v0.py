#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


def ff(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def objective_from_labels(labels, outcome):
    approach = bool(labels.get("approach_success", False))
    stable = bool(labels.get("stable_reached", False))
    drift = bool(labels.get("drift_after_approach", False))
    yaw_sat = bool(labels.get("yaw_saturated", False))

    final_dist = ff(outcome.get("final_rel_dist"), 999.0)
    dx = ff(outcome.get("odom_x_delta"), 0.0)

    # β = [velocity, stability, energy/effort]
    # Current sponge evidence says: reduce speed objective, prioritize stability/hold.
    if stable:
        beta = {
            "beta_velocity": 0.30,
            "beta_stability": 0.55,
            "beta_energy": 0.15,
        }
        semantic = "stable_soft_terrain_locomotion"
        recovery_needed = False

    elif approach and drift:
        beta = {
            "beta_velocity": 0.08,
            "beta_stability": 0.82,
            "beta_energy": 0.10,
        }
        semantic = "approach_possible_but_post_reach_hold_needed"
        recovery_needed = True

    elif final_dist > 0.75 or abs(dx) > 1.0:
        beta = {
            "beta_velocity": 0.05,
            "beta_stability": 0.85,
            "beta_energy": 0.10,
        }
        semantic = "forward_walk_unreliable_on_soft_terrain"
        recovery_needed = True

    else:
        beta = {
            "beta_velocity": 0.12,
            "beta_stability": 0.76,
            "beta_energy": 0.12,
        }
        semantic = "cautious_probe_required"
        recovery_needed = True

    if yaw_sat:
        beta["beta_stability"] = min(0.90, beta["beta_stability"] + 0.05)
        beta["beta_velocity"] = max(0.03, beta["beta_velocity"] - 0.03)

    # Renormalize.
    z = beta["beta_velocity"] + beta["beta_stability"] + beta["beta_energy"]
    for k in beta:
        beta[k] /= z

    return beta, semantic, recovery_needed


def risk_targets(labels, outcome):
    approach = bool(labels.get("approach_success", False))
    stable = bool(labels.get("stable_reached", False))
    drift = bool(labels.get("drift_after_approach", False))
    yaw_sat = bool(labels.get("yaw_saturated", False))

    final_dist = ff(outcome.get("final_rel_dist"), 999.0)
    min_dist = ff(outcome.get("min_rel_dist"), 999.0)
    dx = abs(ff(outcome.get("odom_x_delta"), 0.0))
    yaw = ff(outcome.get("max_abs_mpc_yaw_rate"), 0.0)

    future_invalid = (not stable) and (final_dist > 0.50 or dx > 1.0)
    recovery_needed = drift or future_invalid or yaw_sat
    future_risk = 0.0

    future_risk += 0.35 if not stable else 0.0
    future_risk += 0.20 if drift else 0.0
    future_risk += 0.20 if yaw_sat else 0.0
    future_risk += min(0.20, max(final_dist - 0.30, 0.0) * 0.20)
    future_risk += min(0.15, max(dx - 0.75, 0.0) * 0.10)
    future_risk = max(0.0, min(1.0, future_risk))

    return {
        "approach_success": approach,
        "stable_reached": stable,
        "drift_after_approach": drift,
        "yaw_saturated": yaw_sat,
        "future_invalid": future_invalid,
        "recovery_needed": recovery_needed,
        "future_risk": future_risk,
        "min_rel_dist": min_dist,
        "final_rel_dist": final_dist,
        "abs_odom_x_delta": dx,
        "max_abs_yaw_rate": yaw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sponge-outcome-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.sponge_outcome_jsonl) if line.strip()]
    examples = []

    for r in rows:
        labels = r["labels"]
        outcome = r["outcome"]
        action = r["action"]

        beta, semantic, recovery_needed = objective_from_labels(labels, outcome)
        ram = risk_targets(labels, outcome)

        ex = {
            "dataset_type": "phase_b_sponge_objective_ram_targets_v0",
            "world_name": r.get("world_name"),
            "case_name": r.get("case_name"),
            "obs": {
                "terrain": {
                    "world_name": r.get("world_name"),
                    "terrain_family": "sponge",
                    "soft_contact": True,
                },
                "candidate_action": action,
            },
            "objective_target": {
                **beta,
                "semantic": semantic,
                "recovery_needed": recovery_needed,
            },
            "ram_target": ram,
            "outcome": outcome,
            "labels": labels,
            "outcome_score": r.get("outcome_score"),
            "paths": r.get("paths", {}),
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
        "num_recovery_needed": sum(1 for e in examples if e["objective_target"]["recovery_needed"]),
        "num_stable_reached": sum(1 for e in examples if e["ram_target"]["stable_reached"]),
        "num_approach_success": sum(1 for e in examples if e["ram_target"]["approach_success"]),
        "num_drift_after_approach": sum(1 for e in examples if e["ram_target"]["drift_after_approach"]),
        "out_jsonl": args.out_jsonl,
        "source_jsonl": args.sponge_outcome_jsonl,
    }

    with open(args.out_summary_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
