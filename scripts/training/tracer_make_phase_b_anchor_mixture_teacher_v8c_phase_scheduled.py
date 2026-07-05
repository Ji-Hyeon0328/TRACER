#!/usr/bin/env python3
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.runtime.tracer_run_phase_b_training_campaign_v1 import build_action_bank

TEACHER_IN = "configs/phase_b_anchor_mixture_v8/ppo_warmstart_teacher_table_anchor_mixture_v8b_sweep.json"
TEACHER_OUT = "configs/phase_b_anchor_mixture_v8/ppo_warmstart_teacher_table_anchor_mixture_v8c_phase_scheduled.json"
WORLD = "tracer_sponge_firm_flat"

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def project(t):
    t["vx_far"] = max(0.0, min(0.30, float(t["vx_far"])))
    t["vx_near"] = max(0.0, min(t["vx_far"], float(t["vx_near"])))
    t["goal_stop_distance"] = max(0.10, min(0.30, float(t["goal_stop_distance"])))
    t["goal_slow_distance"] = max(t["goal_stop_distance"] + 0.05, min(0.85, float(t["goal_slow_distance"])))
    t["body_height"] = max(0.28, min(0.36, float(t["body_height"])))
    t["swing_clearance"] = max(0.02, min(0.12, float(t["swing_clearance"])))
    return t

def entry(name, theta, source):
    return {
        "action_name": name,
        "theta_action": project(dict(theta)),
        "source": source,
    }

def main():
    data = load_json(TEACHER_IN)
    bank = build_action_bank(TEACHER_IN)

    world_obj = data["worlds"][WORLD]
    world_obj.setdefault("actions", [])

    names = {
        "sponge_v8c_far_fast_early_stop",
        "sponge_v8c_mid_fast_early_stop",
        "sponge_v8c_soft_stop",
    }

    world_obj["actions"] = [
        a for a in world_obj["actions"]
        if not (isinstance(a, dict) and a.get("action_name") in names)
    ]

    # Start from useful anchors.
    reach_fast = dict(bank["sponge_v8b_reach_bias_fast"])
    soft = dict(bank["trot_soft_mid_clear"])

    # 1) Keep far reach, but brake much earlier.
    theta1 = dict(reach_fast)
    theta1.update({
        "vx_near": 0.0,
        "goal_slow_distance": 0.60,
        "goal_stop_distance": 0.22,
        "body_height": 0.325,
        "swing_clearance": 0.065,
    })

    # 2) Slightly reduce far speed, brake early.
    theta2 = dict(reach_fast)
    theta2.update({
        "vx_far": max(0.0, float(reach_fast["vx_far"]) - 0.015),
        "vx_near": 0.0,
        "goal_slow_distance": 0.62,
        "goal_stop_distance": 0.24,
        "body_height": 0.320,
        "swing_clearance": 0.060,
    })

    # 3) Stability-biased stop behavior, but with enough far progress.
    theta3 = dict(soft)
    theta3.update({
        "vx_far": max(float(soft["vx_far"]), 0.065),
        "vx_near": 0.0,
        "goal_slow_distance": 0.65,
        "goal_stop_distance": 0.25,
        "body_height": 0.315,
        "swing_clearance": 0.060,
    })

    variants = [
        entry("sponge_v8c_far_fast_early_stop", theta1, "v8c_phase_scheduled_far_fast_early_stop"),
        entry("sponge_v8c_mid_fast_early_stop", theta2, "v8c_phase_scheduled_mid_fast_early_stop"),
        entry("sponge_v8c_soft_stop", theta3, "v8c_phase_scheduled_soft_stop"),
    ]

    world_obj["actions"].extend(variants)
    save_json(TEACHER_OUT, data)

    new_bank = build_action_bank(TEACHER_OUT)
    print("[wrote]", TEACHER_OUT)
    for v in variants:
        name = v["action_name"]
        print()
        print(name, "readable:", name in new_bank)
        print(new_bank.get(name))

if __name__ == "__main__":
    main()
