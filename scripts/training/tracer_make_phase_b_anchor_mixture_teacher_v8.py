#!/usr/bin/env python3
import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.runtime.tracer_run_phase_b_training_campaign_v1 import build_action_bank


THETA_KEYS = [
    "vx_far",
    "vx_near",
    "goal_slow_distance",
    "goal_stop_distance",
    "body_height",
    "swing_clearance",
]


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def softmax(xs, temp=1.0):
    m = max(xs)
    ex = [math.exp((x - m) / temp) for x in xs]
    s = sum(ex)
    return [v / s for v in ex]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-in", default="configs/phase_b_training_pipeline_v2_sponge_expanded/ppo_warmstart_teacher_table_sponge_expanded_v0.json")
    ap.add_argument("--teacher-out", default="configs/phase_b_anchor_mixture_v8/ppo_warmstart_teacher_table_anchor_mixture_v8.json")
    ap.add_argument("--world-name", default="tracer_sponge_firm_flat")
    ap.add_argument("--new-action", default="sponge_v8_anchor_mix_top3")
    ap.add_argument("--anchors", default="sponge_probe_crawlish,sponge_reach_then_brake,trot_soft_mid_clear")
    ap.add_argument("--scores", default="0.688,0.677,0.660")
    ap.add_argument("--temperature", type=float, default=0.20)
    args = ap.parse_args()

    original = load_json(args.teacher_in)
    bank = build_action_bank(args.teacher_in)

    anchors = [x.strip() for x in args.anchors.split(",") if x.strip()]
    scores = [float(x.strip()) for x in args.scores.split(",") if x.strip()]
    weights = softmax(scores, temp=args.temperature)

    print("available actions:", sorted(bank.keys()))
    print("anchors:", anchors)
    print("scores:", scores)
    print("weights:", [round(w, 4) for w in weights])

    missing = [a for a in anchors if a not in bank]
    if missing:
        raise KeyError(f"Missing anchors in runtime action bank: {missing}")

    mixed = {}
    for k in THETA_KEYS:
        vals = []
        for a in anchors:
            if k not in bank[a]:
                raise KeyError(f"Anchor {a} does not have key {k}. Available keys: {sorted(bank[a].keys())}")
            vals.append(float(bank[a][k]))
        mixed[k] = sum(w * v for w, v in zip(weights, vals))

    # Safety projection.
    mixed["vx_far"] = max(0.0, min(0.30, mixed["vx_far"]))
    mixed["vx_near"] = max(0.0, min(mixed["vx_far"], mixed["vx_near"]))
    mixed["goal_stop_distance"] = max(0.10, min(0.25, mixed["goal_stop_distance"]))
    mixed["goal_slow_distance"] = max(mixed["goal_stop_distance"] + 0.05, min(0.80, mixed["goal_slow_distance"]))
    mixed["body_height"] = max(0.28, min(0.36, mixed["body_height"]))
    mixed["swing_clearance"] = max(0.02, min(0.12, mixed["swing_clearance"]))

    out = copy.deepcopy(original)

    if "worlds" not in out or not isinstance(out["worlds"], dict):
        raise RuntimeError("Expected teacher json to have dict field `worlds`.")

    if args.world_name not in out["worlds"]:
        raise KeyError(f"world_name not found in teacher json worlds: {args.world_name}")

    world_obj = out["worlds"][args.world_name]
    if not isinstance(world_obj, dict):
        raise RuntimeError(f"world object for {args.world_name} is not dict-like.")

    # Remove stale action with same name from known lists.
    for key in ["actions", "ranked_actions", "ppo_action_prior"]:
        if isinstance(world_obj.get(key), list):
            world_obj[key] = [
                a for a in world_obj[key]
                if not (
                    isinstance(a, dict)
                    and (a.get("action_name") == args.new_action or a.get("name") == args.new_action)
                )
            ]

    # This format is exactly what build_action_bank() understands:
    # name = action_name, theta = theta_action.
    new_entry = {
        "action_name": args.new_action,
        "theta_action": dict(mixed),
        "source": "v8_anchor_mixture_top3",
        "anchors": anchors,
        "anchor_scores": scores,
        "anchor_weights": weights,
    }

    world_obj.setdefault("actions", [])
    if not isinstance(world_obj["actions"], list):
        raise RuntimeError(f"`worlds.{args.world_name}.actions` exists but is not a list.")
    world_obj["actions"].append(new_entry)

    # Optional: set as top action only if none exists. Do not overwrite existing teacher top.
    world_obj.setdefault("anchor_mixture_v8_meta", {})
    world_obj["anchor_mixture_v8_meta"][args.new_action] = {
        "anchors": anchors,
        "anchor_scores": scores,
        "anchor_weights": weights,
        "theta_action": mixed,
    }

    save_json(args.teacher_out, out)

    # Verify output can be read by the same runtime function.
    new_bank = build_action_bank(args.teacher_out)
    if args.new_action not in new_bank:
        raise RuntimeError(f"New action {args.new_action} was not readable by build_action_bank after writing.")

    print("[wrote]", args.teacher_out)
    print("new_action:", args.new_action)
    print("mixed theta:")
    for k in THETA_KEYS:
        print(" ", k, "=", round(float(new_bank[args.new_action][k]), 6))


if __name__ == "__main__":
    main()
