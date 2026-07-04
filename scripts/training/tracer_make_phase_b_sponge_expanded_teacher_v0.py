#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path


SPONGE_ACTIONS = [
    ("sponge_slow_high_clear",       0.060, 0.015, 0.45, 0.15, 0.350, 0.090),
    ("sponge_short_step_stable",     0.045, 0.010, 0.40, 0.15, 0.335, 0.065),
    ("sponge_reach_then_brake",      0.090, 0.000, 0.50, 0.15, 0.340, 0.075),
    ("sponge_mid_brake_clear",      0.075, 0.005, 0.55, 0.15, 0.345, 0.085),
    ("sponge_probe_crawlish",       0.030, 0.000, 0.35, 0.15, 0.310, 0.055),
]


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", default="configs/phase_b_ppo_warmstart_v0/ppo_warmstart_teacher_table.json")
    ap.add_argument("--out-json", default="configs/phase_b_training_pipeline_v2_sponge_expanded/ppo_warmstart_teacher_table_sponge_expanded_v0.json")
    ap.add_argument("--world", default="tracer_sponge_firm_flat")
    args = ap.parse_args()

    src = load_json(args.in_json)
    out = copy.deepcopy(src)

    w = out.setdefault("worlds", {}).setdefault(args.world, {})
    actions = w.setdefault("actions", [])

    existing = set()
    for key in ["ranked_actions", "ppo_action_prior", "actions"]:
        for a in w.get(key, []) or []:
            name = a.get("action_name") or a.get("name")
            if name:
                existing.add(name)

    added = []
    for name, vx_far, vx_near, slow, stop, body_h, clear in SPONGE_ACTIONS:
        if name in existing:
            continue
        actions.append({
            "action_name": name,
            "theta_action": {
                "vx_far": vx_far,
                "vx_near": vx_near,
                "goal_slow_distance": slow,
                "goal_stop_distance": stop,
                "body_height": body_h,
                "swing_clearance": clear,
            },
            "source": "sponge_expanded_v0",
            "note": "Sponge-specific theta-lite candidate for soft-contact terrain."
        })
        added.append(name)

    out["sponge_expanded_v0_meta"] = {
        "base_teacher_json": args.in_json,
        "expanded_world": args.world,
        "added_actions": added,
        "num_added": len(added),
        "note": "Sponge-specific theta-lite candidates for action-bank expansion."
    }

    save_json(Path(args.out_json), out)
    print(json.dumps({"out_json": args.out_json, "added": added}, indent=2))


if __name__ == "__main__":
    main()
