#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path


BAD_SEMANTICS = {
    "forward_walk_unreliable_on_soft_terrain",
    "no_meaningful_progress",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-jsonl", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.dataset_jsonl) if line.strip()]
    by_world = defaultdict(list)
    for r in rows:
        by_world[r["world_name"]].append(r)

    table = {}
    for world, rs in by_world.items():
        ranked = sorted(rs, key=lambda r: r["label"]["reward"], reverse=True)
        best = ranked[0]
        sem = best["label"]["semantic"]
        blocked = sem in BAD_SEMANTICS

        table[world] = {
            "world_name": world,
            "selected_profile_name": best["profile_name"],
            "selected_profile_family": best.get("profile_family"),
            "theta_action": best["theta_action"],
            "reward": best["label"]["reward"],
            "semantic": sem,
            "normal_walk_blocked": blocked,
            "requires_alternative_primitive": blocked,
            "metrics": best["metrics"],
            "num_candidates": len(rs),
            "ranked_profiles": [
                {
                    "profile_name": r["profile_name"],
                    "reward": r["label"]["reward"],
                    "semantic": r["label"]["semantic"],
                    "theta_action": r["theta_action"],
                    "metrics": r["metrics"],
                }
                for r in ranked
            ],
        }

    out = {
        "schema": "phase_b_theta_lite_teacher_table_v0",
        "source_dataset_jsonl": args.dataset_jsonl,
        "num_worlds": len(table),
        "table": table,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
