#!/usr/bin/env python3
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.runtime.tracer_run_phase_b_training_campaign_v1 import build_action_bank


TEACHER_IN = "configs/phase_b_anchor_mixture_v8/ppo_warmstart_teacher_table_anchor_mixture_v8.json"
TEACHER_OUT = "configs/phase_b_anchor_mixture_v8/ppo_warmstart_teacher_table_anchor_mixture_v8b_sweep.json"
WORLD = "tracer_sponge_firm_flat"

THETA_KEYS = [
    "vx_far",
    "vx_near",
    "goal_slow_distance",
    "goal_stop_distance",
    "body_height",
    "swing_clearance",
]

ANCHORS = [
    "sponge_probe_crawlish",
    "sponge_reach_then_brake",
    "trot_soft_mid_clear",
]


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def project(theta):
    theta["vx_far"] = max(0.0, min(0.30, float(theta["vx_far"])))
    theta["vx_near"] = max(0.0, min(theta["vx_far"], float(theta["vx_near"])))
    theta["goal_stop_distance"] = max(0.10, min(0.25, float(theta["goal_stop_distance"])))
    theta["goal_slow_distance"] = max(
        theta["goal_stop_distance"] + 0.05,
        min(0.80, float(theta["goal_slow_distance"])),
    )
    theta["body_height"] = max(0.28, min(0.36, float(theta["body_height"])))
    theta["swing_clearance"] = max(0.02, min(0.12, float(theta["swing_clearance"])))
    return theta


def mix_theta(bank, weights):
    theta = {}
    for k in THETA_KEYS:
        theta[k] = sum(float(weights[a]) * float(bank[a][k]) for a in ANCHORS)
    return project(theta)


def apply_overrides(theta, overrides):
    theta = dict(theta)
    for k, v in overrides.items():
        if k == "delta_vx_far":
            theta["vx_far"] += float(v)
        elif k == "delta_vx_near":
            theta["vx_near"] += float(v)
        elif k == "delta_clearance":
            theta["swing_clearance"] += float(v)
        elif k == "delta_body_height":
            theta["body_height"] += float(v)
        else:
            theta[k] = float(v)
    return project(theta)


def make_entry(name, theta, weights, overrides):
    return {
        "action_name": name,
        "theta_action": dict(theta),
        "source": "v8b_anchor_mixture_sweep",
        "anchors": ANCHORS,
        "anchor_weights": weights,
        "overrides": overrides,
    }


def main():
    data = load_json(TEACHER_IN)
    bank = build_action_bank(TEACHER_IN)

    print("available actions:", sorted(bank.keys()))
    for a in ANCHORS:
        if a not in bank:
            raise KeyError(f"missing anchor: {a}")

    variants = [
        {
            "name": "sponge_v8b_reach_bias",
            "weights": {
                "sponge_probe_crawlish": 0.15,
                "sponge_reach_then_brake": 0.60,
                "trot_soft_mid_clear": 0.25,
            },
            "overrides": {},
        },
        {
            "name": "sponge_v8b_reach_bias_fast",
            "weights": {
                "sponge_probe_crawlish": 0.15,
                "sponge_reach_then_brake": 0.60,
                "trot_soft_mid_clear": 0.25,
            },
            "overrides": {
                "delta_vx_far": 0.015,
                "vx_near": 0.005,
                "goal_slow_distance": 0.35,
            },
        },
        {
            "name": "sponge_v8b_soft_reach",
            "weights": {
                "sponge_probe_crawlish": 0.10,
                "sponge_reach_then_brake": 0.45,
                "trot_soft_mid_clear": 0.45,
            },
            "overrides": {
                "vx_near": 0.005,
                "goal_slow_distance": 0.35,
            },
        },
        {
            "name": "sponge_v8b_probe_reach_fast",
            "weights": {
                "sponge_probe_crawlish": 0.25,
                "sponge_reach_then_brake": 0.55,
                "trot_soft_mid_clear": 0.20,
            },
            "overrides": {
                "delta_vx_far": 0.020,
                "vx_near": 0.005,
                "goal_slow_distance": 0.34,
            },
        },
        {
            "name": "sponge_v8b_reach_stabilized",
            "weights": {
                "sponge_probe_crawlish": 0.20,
                "sponge_reach_then_brake": 0.70,
                "trot_soft_mid_clear": 0.10,
            },
            "overrides": {
                "vx_near": 0.000,
                "goal_slow_distance": 0.38,
                "swing_clearance": 0.065,
                "body_height": 0.325,
            },
        },
    ]

    out = copy.deepcopy(data)
    world_obj = out["worlds"][WORLD]
    world_obj.setdefault("actions", [])

    # Remove stale v8b variants.
    names = {v["name"] for v in variants}
    world_obj["actions"] = [
        a for a in world_obj["actions"]
        if not (
            isinstance(a, dict)
            and (a.get("action_name") in names or a.get("name") in names)
        )
    ]

    for v in variants:
        theta = mix_theta(bank, v["weights"])
        theta = apply_overrides(theta, v["overrides"])
        entry = make_entry(v["name"], theta, v["weights"], v["overrides"])
        world_obj["actions"].append(entry)

        print()
        print(v["name"])
        print(" weights:", v["weights"])
        print(" overrides:", v["overrides"])
        print(" theta:", {k: round(theta[k], 6) for k in THETA_KEYS})

    save_json(TEACHER_OUT, out)

    new_bank = build_action_bank(TEACHER_OUT)
    print()
    print("[wrote]", TEACHER_OUT)
    for v in variants:
        print(v["name"], "readable:", v["name"] in new_bank, new_bank.get(v["name"]))


if __name__ == "__main__":
    main()
