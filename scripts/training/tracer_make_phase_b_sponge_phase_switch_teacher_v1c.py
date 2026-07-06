#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.runtime.tracer_run_phase_b_training_campaign_v1 import build_action_bank

TEACHER_IN = "configs/phase_b_anchor_mixture_v8/ppo_warmstart_teacher_table_anchor_mixture_v8c_phase_scheduled.json"
TEACHER_OUT = "configs/phase_b_sponge_phase_switch_v1c/ppo_warmstart_teacher_table_sponge_phase_switch_v1c.json"

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
    t["vx_far"] = max(0.0, min(0.30, t["vx_far"]))
    t["vx_near"] = max(0.0, min(t["vx_far"], t["vx_near"]))
    t["goal_stop_distance"] = max(0.10, min(0.30, t["goal_stop_distance"]))
    t["goal_slow_distance"] = max(t["goal_stop_distance"] + 0.05, min(0.85, t["goal_slow_distance"]))
    t["body_height"] = max(0.28, min(0.36, t["body_height"]))
    t["swing_clearance"] = max(0.02, min(0.12, t["swing_clearance"]))
    return t

def add_action(data, name, theta_action, source_note):
    w = data["worlds"]["tracer_sponge_firm_flat"]
    w.setdefault("actions", [])
    w["actions"] = [
        a for a in w["actions"]
        if not (isinstance(a, dict) and a.get("action_name") == name)
    ]
    w["actions"].append({
        "action_name": name,
        "theta_action": theta_action,
        "source": "sponge_phase_switch_v1c",
        "source_note": source_note,
    })

def mix(a, b, wa):
    return {k: wa * a[k] + (1.0 - wa) * b[k] for k in THETA_KEYS}

def main():
    data = load_json(TEACHER_IN)
    bank = build_action_bank(TEACHER_IN)

    reach_stabilized = theta(bank, "sponge_v8b_reach_stabilized")
    reach_bias = theta(bank, "sponge_v8b_reach_bias")
    hold = theta(bank, "sponge_slow_high_clear")

    # 1) Far은 reach_stabilized, near/stop은 hold를 강하게 반영.
    a = {
        "vx_far": reach_stabilized["vx_far"],
        "vx_near": hold["vx_near"],
        "goal_slow_distance": max(hold["goal_slow_distance"], 0.55),
        "goal_stop_distance": max(hold["goal_stop_distance"], 0.22),
        "body_height": 0.5 * reach_stabilized["body_height"] + 0.5 * hold["body_height"],
        "swing_clearance": 0.5 * reach_stabilized["swing_clearance"] + 0.5 * hold["swing_clearance"],
    }
    add_action(
        data,
        "sponge_v1c_reach_stabilized_to_hold",
        project(a),
        "far velocity from sponge_v8b_reach_stabilized, near/stop from sponge_slow_high_clear",
    )

    # 2) Far은 reach_bias, near/stop은 hold.
    b = {
        "vx_far": reach_bias["vx_far"],
        "vx_near": hold["vx_near"],
        "goal_slow_distance": max(hold["goal_slow_distance"], 0.55),
        "goal_stop_distance": max(hold["goal_stop_distance"], 0.22),
        "body_height": 0.5 * reach_bias["body_height"] + 0.5 * hold["body_height"],
        "swing_clearance": 0.5 * reach_bias["swing_clearance"] + 0.5 * hold["swing_clearance"],
    }
    add_action(
        data,
        "sponge_v1c_reach_bias_to_hold",
        project(b),
        "far velocity from sponge_v8b_reach_bias, near/stop from sponge_slow_high_clear",
    )

    # 3) Conservative hybrid: reach_bias와 hold를 직접 평균하되 stop을 크게 둠.
    c = mix(reach_bias, hold, 0.55)
    c["goal_slow_distance"] = max(c["goal_slow_distance"], 0.60)
    c["goal_stop_distance"] = max(c["goal_stop_distance"], 0.23)
    c["vx_near"] = min(c["vx_near"], 0.005)
    add_action(
        data,
        "sponge_v1c_conservative_reach_hold",
        project(c),
        "conservative interpolation between sponge_v8b_reach_bias and sponge_slow_high_clear",
    )

    save_json(TEACHER_OUT, data)

    new_bank = build_action_bank(TEACHER_OUT)
    print("[wrote]", TEACHER_OUT)
    for n in [
        "sponge_v1c_reach_stabilized_to_hold",
        "sponge_v1c_reach_bias_to_hold",
        "sponge_v1c_conservative_reach_hold",
        "sponge_v8b_reach_stabilized",
        "sponge_v8b_reach_bias",
        "sponge_slow_high_clear",
    ]:
        print()
        print(n, n in new_bank)
        t = new_bank[n]
        for k in THETA_KEYS:
            print(f"  {k}: {float(t[k]):.6f}")

if __name__ == "__main__":
    main()
