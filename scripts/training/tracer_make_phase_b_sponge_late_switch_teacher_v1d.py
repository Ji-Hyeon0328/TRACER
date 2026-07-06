#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.runtime.tracer_run_phase_b_training_campaign_v1 import build_action_bank

TEACHER_IN = "configs/phase_b_anchor_mixture_v8/ppo_warmstart_teacher_table_anchor_mixture_v8c_phase_scheduled.json"
TEACHER_OUT = "configs/phase_b_sponge_late_switch_v1d/ppo_warmstart_teacher_table_sponge_late_switch_v1d.json"

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

def theta(bank, name):
    t = bank[name]
    return {k: float(t[k]) for k in THETA_KEYS}

def project(t):
    t["vx_far"] = max(0.0, min(0.30, float(t["vx_far"])))
    t["vx_near"] = max(0.0, min(t["vx_far"], float(t["vx_near"])))
    t["goal_stop_distance"] = max(0.10, min(0.30, float(t["goal_stop_distance"])))
    t["goal_slow_distance"] = max(t["goal_stop_distance"] + 0.05, min(0.85, float(t["goal_slow_distance"])))
    t["body_height"] = max(0.28, min(0.36, float(t["body_height"])))
    t["swing_clearance"] = max(0.02, min(0.12, float(t["swing_clearance"])))
    return t

def add_action(data, name, theta_action, note):
    w = data["worlds"]["tracer_sponge_firm_flat"]
    w.setdefault("actions", [])
    w["actions"] = [
        a for a in w["actions"]
        if not (isinstance(a, dict) and a.get("action_name") == name)
    ]
    w["actions"].append({
        "action_name": name,
        "theta_action": theta_action,
        "source": "sponge_late_switch_v1d",
        "source_note": note,
    })

def make_profile(reach, hold, slow, stop, vx_near_scale, name):
    # Far는 reach target에서 가져오고,
    # near/height/clearance는 hold target 쪽을 강하게 반영한다.
    return project({
        "vx_far": reach["vx_far"],
        "vx_near": max(0.0, hold["vx_near"] * vx_near_scale),
        "goal_slow_distance": slow,
        "goal_stop_distance": stop,
        "body_height": 0.35 * reach["body_height"] + 0.65 * hold["body_height"],
        "swing_clearance": 0.35 * reach["swing_clearance"] + 0.65 * hold["swing_clearance"],
    })

def main():
    data = load_json(TEACHER_IN)
    bank = build_action_bank(TEACHER_IN)

    reach_bias = theta(bank, "sponge_v8b_reach_bias")
    reach_stabilized = theta(bank, "sponge_v8b_reach_stabilized")
    hold = theta(bank, "sponge_slow_high_clear")

    candidates = [
        (
            "sponge_v1d_bias_late_hold_025_013",
            reach_bias,
            0.25,
            0.13,
            1.00,
            "reach_bias far; late slow=0.25 stop=0.13; hold near",
        ),
        (
            "sponge_v1d_bias_late_hold_030_015",
            reach_bias,
            0.30,
            0.15,
            1.00,
            "reach_bias far; late slow=0.30 stop=0.15; hold near",
        ),
        (
            "sponge_v1d_bias_late_hold_035_016",
            reach_bias,
            0.35,
            0.16,
            1.00,
            "reach_bias far; late slow=0.35 stop=0.16; hold near",
        ),
        (
            "sponge_v1d_stabilized_late_hold_025_013",
            reach_stabilized,
            0.25,
            0.13,
            1.00,
            "reach_stabilized far; late slow=0.25 stop=0.13; hold near",
        ),
        (
            "sponge_v1d_stabilized_late_hold_030_015",
            reach_stabilized,
            0.30,
            0.15,
            1.00,
            "reach_stabilized far; late slow=0.30 stop=0.15; hold near",
        ),
        (
            "sponge_v1d_stabilized_late_hold_035_016",
            reach_stabilized,
            0.35,
            0.16,
            1.00,
            "reach_stabilized far; late slow=0.35 stop=0.16; hold near",
        ),
    ]

    for name, reach, slow, stop, near_scale, note in candidates:
        add_action(data, name, make_profile(reach, hold, slow, stop, near_scale, name), note)

    save_json(TEACHER_OUT, data)

    new_bank = build_action_bank(TEACHER_OUT)
    print("[wrote]", TEACHER_OUT)
    for name, *_ in candidates:
        print()
        print(name, name in new_bank)
        t = new_bank[name]
        for k in THETA_KEYS:
            print(f"  {k}: {float(t[k]):.6f}")

if __name__ == "__main__":
    main()
