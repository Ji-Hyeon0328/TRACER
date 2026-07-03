#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


def infer_route(row):
    label = row.get("label", {})
    world = row.get("world_name", "")

    rollout_executed = bool(label.get("rollout_executed", False))
    normal_walk_blocked = bool(label.get("normal_walk_blocked", False))
    requires_alt = bool(label.get("requires_alternative_primitive", False))
    requires_more = bool(label.get("requires_more_theta_search", False))
    stable = bool(label.get("stable_reached", False))

    if rollout_executed and stable and not normal_walk_blocked:
        return {
            "route_label": "theta_lite_diagonal_normal_walk",
            "allow_theta_lite_diagonal": True,
            "requires_alternative_primitive": False,
            "requires_more_theta_search": False,
            "target_gait_family": "diagonal_trot_theta_lite",
            "target_primitive_family": "normal_diagonal_locomotion",
            "reason": "trusted theta-lite diagonal gait succeeded",
        }

    if normal_walk_blocked and requires_alt:
        # Current Phase-B cannot execute this yet; this is the target for the
        # next unified GMS/RL meta-gait planner action-space expansion.
        if "sponge" in world and "down" in world:
            candidates = [
                "soft_contact_downslope_controlled_descent",
                "backstep_or_quasi_static_descent",
                "high_support_soft_contact_crawl",
            ]
        elif "sponge" in world and "slope" in world:
            candidates = [
                "soft_contact_upslope_crawl",
                "high_support_high_clearance_walk",
                "terrain_compliance_adaptive_gait",
            ]
        elif "sponge" in world:
            candidates = [
                "soft_contact_crawl_or_quasi_static",
                "high_support_high_clearance_walk",
                "post_reach_hold_and_brake",
            ]
        else:
            candidates = [
                "alternative_primitive_required",
            ]

        return {
            "route_label": "alternative_primitive_required",
            "allow_theta_lite_diagonal": False,
            "requires_alternative_primitive": True,
            "requires_more_theta_search": False,
            "target_gait_family": "not_theta_lite_diagonal",
            "target_primitive_family": candidates[0],
            "candidate_primitive_families": candidates,
            "reason": "robust theta-lite table blocked normal diagonal gait",
        }

    if normal_walk_blocked and requires_more:
        return {
            "route_label": "theta_lite_more_search_required",
            "allow_theta_lite_diagonal": False,
            "requires_alternative_primitive": False,
            "requires_more_theta_search": True,
            "target_gait_family": "theta_lite_local_search",
            "target_primitive_family": "defer_for_more_theta_lite_search",
            "reason": "table did not trust normal walking but did not require alternative primitive",
        }

    return {
        "route_label": "untrusted_or_unknown",
        "allow_theta_lite_diagonal": False,
        "requires_alternative_primitive": False,
        "requires_more_theta_search": True,
        "target_gait_family": "unknown",
        "target_primitive_family": "unknown",
        "reason": "insufficient or inconsistent guarded outcome label",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guarded-outcome-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.guarded_outcome_jsonl) if line.strip()]

    out_rows = []
    for r in rows:
        route = infer_route(r)
        out_rows.append({
            "schema": "phase_b_primitive_route_dataset_row_v0",
            "source_guarded_outcome_jsonl": args.guarded_outcome_jsonl,
            "source_run_dir": r.get("source_run_dir"),
            "world_name": r.get("world_name"),
            "selected_profile_name": r.get("selected_profile_name"),
            "selection_status": r.get("selection_status"),
            "theta_action": r.get("theta_action", {}),
            "guarded_label": r.get("label", {}),
            "metrics": r.get("metrics", {}),
            "route_target": route,
            "features_v0": {
                # For now this is symbolic/contextual. Later this should be replaced
                # or augmented with RAM rho/sigma, Objective beta, terrain encoder c_t,
                # and proprio history.
                "world_name": r.get("world_name"),
                "selection_status": r.get("selection_status"),
                "normal_walk_blocked": bool(r.get("label", {}).get("normal_walk_blocked", False)),
                "rollout_executed": bool(r.get("label", {}).get("rollout_executed", False)),
                "theta_vx_far": r.get("theta_action", {}).get("vx_far"),
                "theta_vx_near": r.get("theta_action", {}).get("vx_near"),
                "theta_body_height": r.get("theta_action", {}).get("body_height"),
                "theta_swing_clearance": r.get("theta_action", {}).get("swing_clearance"),
            },
        })

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    summary = {
        "schema": "phase_b_primitive_route_dataset_summary_v0",
        "source_guarded_outcome_jsonl": args.guarded_outcome_jsonl,
        "num_examples": len(out_rows),
        "route_counts": dict(Counter(r["route_target"]["route_label"] for r in out_rows)),
        "world_counts": dict(Counter(r["world_name"] for r in out_rows)),
        "target_primitive_counts": dict(Counter(r["route_target"]["target_primitive_family"] for r in out_rows)),
        "out_jsonl": args.out_jsonl,
    }

    Path(args.out_summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
