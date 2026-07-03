#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-jsonl", required=True)
    ap.add_argument("--seed-json", default="configs/phase_b_primitive_theta_extended_v0/primitive_theta_extended_seed_v0.json")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    args = ap.parse_args()

    seeds = json.load(open(args.seed_json))
    primitive_families = seeds["primitive_families"]

    rows = []
    missing = []

    for line in open(args.route_jsonl):
        if not line.strip():
            continue
        r = json.loads(line)
        target_family = r["route_target"]["target_primitive_family"]

        seed = primitive_families.get(target_family)
        if seed is None:
            missing.append(target_family)
            r["theta_extended_target"] = None
            r["theta_extended_available"] = False
        else:
            r["theta_extended_target"] = {
                "primitive_family": target_family,
                "description": seed.get("description"),
                "executable_by_current_a1_qpmc": seed.get("executable_by_current_a1_qpmc", False),
                "requires_lowlevel_phase_interface": seed.get("requires_lowlevel_phase_interface", True),
                "theta_extended": seed["theta_extended"]
            }
            r["theta_extended_available"] = True

        r["schema"] = "phase_b_primitive_route_with_theta_extended_row_v0"
        rows.append(r)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    summary = {
        "schema": "phase_b_primitive_route_with_theta_extended_summary_v0",
        "source_route_jsonl": args.route_jsonl,
        "seed_json": args.seed_json,
        "num_examples": len(rows),
        "num_theta_extended_available": sum(1 for r in rows if r["theta_extended_available"]),
        "missing_target_families": sorted(set(missing)),
        "route_counts": dict(Counter(r["route_target"]["route_label"] for r in rows)),
        "target_primitive_counts": dict(Counter(r["route_target"]["target_primitive_family"] for r in rows)),
        "executable_by_current_a1_qpmc_counts": dict(Counter(
            str((r.get("theta_extended_target") or {}).get("executable_by_current_a1_qpmc"))
            for r in rows
        )),
        "out_jsonl": args.out_jsonl
    }

    Path(args.out_summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
