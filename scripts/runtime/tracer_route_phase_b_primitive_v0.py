#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def choose_alt_candidates(world):
    if "sponge" in world and "down" in world:
        return [
            "soft_contact_downslope_controlled_descent",
            "backstep_or_quasi_static_descent",
            "high_support_soft_contact_crawl",
        ]
    if "sponge" in world and "slope" in world:
        return [
            "soft_contact_upslope_crawl",
            "high_support_high_clearance_walk",
            "terrain_compliance_adaptive_gait",
        ]
    if "sponge" in world:
        return [
            "soft_contact_crawl_or_quasi_static",
            "high_support_high_clearance_walk",
            "post_reach_hold_and_brake",
        ]
    return ["alternative_primitive_required"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-name", required=True)
    ap.add_argument("--table-json", default="configs/phase_b_theta_lite_robust_teacher_table_v0/current_table.json")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    table = json.load(open(args.table_json))
    entry = table.get("table", {}).get(args.world_name)

    if entry is None:
        out = {
            "schema": "phase_b_primitive_route_v0",
            "world_name": args.world_name,
            "found_in_table": False,
            "route_label": "unknown_requires_more_search",
            "allow_theta_lite_diagonal": False,
            "requires_more_theta_search": True,
            "requires_alternative_primitive": False,
            "reason": "world not found in robust theta-lite table",
        }
    elif entry.get("trusted_normal_walk", False):
        out = {
            "schema": "phase_b_primitive_route_v0",
            "world_name": args.world_name,
            "found_in_table": True,
            "route_label": "theta_lite_diagonal_normal_walk",
            "allow_theta_lite_diagonal": True,
            "requires_more_theta_search": False,
            "requires_alternative_primitive": False,
            "selected_profile_name": entry.get("selected_profile_name"),
            "theta_action": entry.get("theta_action"),
            "target_primitive_family": "normal_diagonal_locomotion",
            "reason": "robust table trusts theta-lite diagonal gait",
        }
    elif entry.get("requires_alternative_primitive", False):
        candidates = choose_alt_candidates(args.world_name)
        out = {
            "schema": "phase_b_primitive_route_v0",
            "world_name": args.world_name,
            "found_in_table": True,
            "route_label": "alternative_primitive_required",
            "allow_theta_lite_diagonal": False,
            "requires_more_theta_search": False,
            "requires_alternative_primitive": True,
            "selected_profile_name": entry.get("selected_profile_name"),
            "blocked_theta_action": entry.get("theta_action"),
            "target_primitive_family": candidates[0],
            "candidate_primitive_families": candidates,
            "reason": "robust table blocks theta-lite diagonal gait",
        }
    else:
        out = {
            "schema": "phase_b_primitive_route_v0",
            "world_name": args.world_name,
            "found_in_table": True,
            "route_label": "theta_lite_more_search_required",
            "allow_theta_lite_diagonal": False,
            "requires_more_theta_search": True,
            "requires_alternative_primitive": False,
            "selected_profile_name": entry.get("selected_profile_name"),
            "theta_action": entry.get("theta_action"),
            "target_primitive_family": "theta_lite_local_search",
            "reason": "robust table does not trust deployment but requests more theta search",
        }

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
