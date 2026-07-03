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


def terrain_family(world_name):
    w = str(world_name or "")
    if "sponge" in w:
        return "sponge"
    if "slippery" in w:
        return "slippery"
    if "rough" in w:
        return "rough"
    if "slope" in w:
        return "slope"
    if w == "earth":
        return "flat"
    return "unknown"


def beta_normalize(beta):
    z = sum(beta.values())
    if z <= 0:
        return beta
    return {k: v / z for k, v in beta.items()}


def make_trusted_positive_example(key, entry):
    world, risk = key.split("::", 1)
    action = entry["action"]
    replay = entry.get("replay_stats", {})

    # Trusted flat locomotion: velocity can matter, but keep stability non-trivial
    # because even earth replay has yaw activity.
    beta = beta_normalize({
        "beta_velocity": 0.45,
        "beta_stability": 0.40,
        "beta_energy": 0.15,
    })

    avg_yaw = ff(replay.get("replay_avg_max_yaw"), 0.0)
    avg_final = ff(replay.get("replay_avg_final_rel_dist"), 0.0)
    avg_dx = abs(ff(replay.get("replay_avg_odom_x_delta"), 0.0))

    future_risk = 0.08
    future_risk += min(0.12, max(avg_yaw - 0.18, 0.0) * 0.6)
    future_risk += min(0.08, max(avg_final - 0.15, 0.0) * 0.4)
    future_risk += min(0.05, max(avg_dx - 0.50, 0.0) * 0.1)
    future_risk = max(0.0, min(1.0, future_risk))

    return {
        "dataset_type": "phase_b_mixed_objective_ram_v0",
        "source_type": "trusted_theta5_table",
        "source_key": key,
        "world_name": world,
        "case_name": entry.get("case_name"),
        "obs": {
            "terrain": {
                "world_name": world,
                "terrain_family": terrain_family(world),
                "soft_contact": "sponge" in world,
            },
            "risk_state_hint": {
                "risk_name": risk,
            },
            "candidate_action": action,
        },
        "objective_target": {
            **beta,
            "semantic": "stable_goal_reach_flat_locomotion",
            "recovery_needed": False,
        },
        "ram_target": {
            "approach_success": True,
            "stable_reached": True,
            "drift_after_approach": False,
            "yaw_saturated": avg_yaw >= 0.299,
            "future_invalid": False,
            "recovery_needed": False,
            "future_risk": future_risk,
            "min_rel_dist": ff(replay.get("replay_avg_min_rel_dist")),
            "final_rel_dist": avg_final,
            "abs_odom_x_delta": avg_dx,
            "max_abs_yaw_rate": avg_yaw,
        },
        "outcome": {
            "replay_stats": replay,
        },
        "labels": {
            "approach_success": True,
            "stable_reached": True,
            "drift_after_approach": False,
            "yaw_saturated": avg_yaw >= 0.299,
        },
        "outcome_score": 100.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sponge-objective-ram-jsonl", required=True)
    ap.add_argument("--trusted-table-json", default="configs/phase_b_theta5_teacher_table_v0/trusted_table.json")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    args = ap.parse_args()

    examples = []

    # Sponge negative/recovery examples.
    with open(args.sponge_objective_ram_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            r["dataset_type"] = "phase_b_mixed_objective_ram_v0"
            r["source_type"] = "sponge_objective_ram_targets"
            examples.append(r)

    # Trusted positive table examples.
    with open(args.trusted_table_json, "r") as f:
        table = json.load(f)

    for key, entry in sorted(table.get("best_table", {}).items()):
        examples.append(make_trusted_positive_example(key, entry))

    semantic_counts = {}
    recovery_counts = {True: 0, False: 0}
    stable_counts = {True: 0, False: 0}

    for ex in examples:
        sem = ex["objective_target"]["semantic"]
        semantic_counts[sem] = semantic_counts.get(sem, 0) + 1
        recovery_counts[bool(ex["objective_target"]["recovery_needed"])] += 1
        stable_counts[bool(ex["ram_target"]["stable_reached"])] += 1

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, sort_keys=True) + "\n")

    summary = {
        "num_examples": len(examples),
        "semantic_counts": semantic_counts,
        "recovery_needed_counts": {
            "true": recovery_counts[True],
            "false": recovery_counts[False],
        },
        "stable_reached_counts": {
            "true": stable_counts[True],
            "false": stable_counts[False],
        },
        "source_sponge_objective_ram_jsonl": args.sponge_objective_ram_jsonl,
        "source_trusted_table_json": args.trusted_table_json,
        "out_jsonl": args.out_jsonl,
    }

    with open(args.out_summary_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
