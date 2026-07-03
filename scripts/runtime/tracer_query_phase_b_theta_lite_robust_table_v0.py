#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


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
            "schema": "phase_b_theta_lite_robust_table_query_v0",
            "world_name": args.world_name,
            "found": False,
            "allow_normal_walk": False,
            "normal_walk_blocked": True,
            "requires_more_theta_search": True,
            "requires_alternative_primitive": False,
            "reason": "world_not_found_in_robust_theta_lite_table",
        }
    else:
        out = {
            "schema": "phase_b_theta_lite_robust_table_query_v0",
            "world_name": args.world_name,
            "found": True,
            "selection_status": entry.get("selection_status"),
            "selected_profile_name": entry.get("selected_profile_name"),
            "theta_action": entry.get("theta_action"),
            "allow_normal_walk": bool(entry.get("trusted_normal_walk", False)),
            "normal_walk_blocked": bool(entry.get("normal_walk_blocked", True)),
            "requires_more_theta_search": bool(entry.get("requires_more_theta_search", False)),
            "requires_alternative_primitive": bool(entry.get("requires_alternative_primitive", False)),
            "selected_profile_summary": entry.get("selected_profile_summary"),
        }

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
