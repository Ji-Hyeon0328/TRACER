#!/usr/bin/env python3
import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

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

WORLD_KEYS = [
    "earth",
    "stairs_single",
    "tracer_sponge_firm_flat",
]

TARGET_MODES = [
    "balanced",
    "reach",
    "stability",
]

ANCHOR_SCALE = {
    "vx_far": 0.30,
    "vx_near": 0.10,
    "goal_slow_distance": 0.85,
    "goal_stop_distance": 0.30,
    "body_height": 0.36,
    "swing_clearance": 0.12,
}

DELTA_SCALE = {
    "vx_far": 0.10,
    "vx_near": 0.05,
    "goal_slow_distance": 0.30,
    "goal_stop_distance": 0.15,
    "body_height": 0.05,
    "swing_clearance": 0.05,
}

WORLD_BETA = {
    "earth": [0.600, 0.230, 0.170],
    "stairs_single": [0.461, 0.399, 0.141],
    "tracer_sponge_firm_flat": [0.140, 0.796, 0.063],
}


class ResidualActorV1(nn.Module):
    def __init__(self, input_dim, output_dim=6, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)


def get_world_action(d):
    world = d.get("world") or d.get("world_name")
    action = d.get("action") or d.get("action_name") or d.get("selected_action_name")
    wa = d.get("world_action") or d.get("world::action")
    if (not world or not action) and isinstance(wa, str) and "::" in wa:
        world, action = wa.split("::", 1)
    return world, action


def extract_task_scores(path):
    obj = load_json(path)
    scores = {}
    for d in walk(obj):
        if not isinstance(d, dict):
            continue
        world, action = get_world_action(d)
        if not world or not action:
            continue
        score = d.get("score_task_beta")
        if score is None:
            score = d.get("target_task_beta")
        if score is None:
            continue
        try:
            scores[(world, action)] = float(score)
        except Exception:
            pass
    return scores


def project_theta(t):
    t["vx_far"] = max(0.0, min(0.30, float(t["vx_far"])))
    t["vx_near"] = max(0.0, min(t["vx_far"], float(t["vx_near"])))
    t["goal_stop_distance"] = max(0.10, min(0.30, float(t["goal_stop_distance"])))
    t["goal_slow_distance"] = max(
        t["goal_stop_distance"] + 0.05,
        min(0.85, float(t["goal_slow_distance"])),
    )
    t["body_height"] = max(0.28, min(0.36, float(t["body_height"])))
    t["swing_clearance"] = max(0.02, min(0.12, float(t["swing_clearance"])))
    return t


def feature_vec(world, mode, beta, theta, anchor_score):
    x = []

    for w in WORLD_KEYS:
        x.append(1.0 if world == w else 0.0)

    for m in TARGET_MODES:
        x.append(1.0 if mode == m else 0.0)

    x.extend(beta)

    for k in THETA_KEYS:
        x.append(float(theta[k]) / ANCHOR_SCALE[k])

    x.append(float(anchor_score))
    return x


def add_action_to_world(data, world, action_name, theta, meta):
    world_obj = data["worlds"][world]
    world_obj.setdefault("actions", [])

    world_obj["actions"] = [
        a for a in world_obj["actions"]
        if not (isinstance(a, dict) and a.get("action_name") == action_name)
    ]

    world_obj["actions"].append({
        "action_name": action_name,
        "theta_action": theta,
        "source": "theta_residual_actor_v1_multimode",
        **meta,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-in", default="configs/phase_b_anchor_mixture_v8/ppo_warmstart_teacher_table_anchor_mixture_v8c_phase_scheduled.json")
    ap.add_argument("--teacher-out", default="configs/phase_b_theta_residual_actor_v1_multimode/ppo_warmstart_teacher_table_theta_residual_actor_v1_multimode.json")
    ap.add_argument("--model", default="artifacts/phase_b_theta_residual_actor_v1_multimode/model.pt")
    ap.add_argument("--task-beta-report", default="reports/phase_b_task_beta_reward_v1_anchor_mixture_merged.json")
    args = ap.parse_args()

    data = load_json(args.teacher_in)
    bank = build_action_bank(args.teacher_in)
    scores = extract_task_scores(args.task_beta_report)

    ckpt = torch.load(args.model, map_location="cpu")
    model = ResidualActorV1(input_dim=int(ckpt["input_dim"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Same-anchor multimode test. This checks whether target_mode truly changes theta.
    specs = [
        ("earth", "trot_solid_fast", "balanced", "theta_actor_v1_earth_balanced_from_trot_solid_fast"),
        ("earth", "trot_mid", "reach", "theta_actor_v1_earth_reach_from_trot_mid"),
        ("earth", "trot_solid_fast", "stability", "theta_actor_v1_earth_stability_from_trot_solid_fast"),

        ("stairs_single", "trot_mid", "balanced", "theta_actor_v1_stairs_balanced_from_trot_mid"),
        ("stairs_single", "trot_mid", "reach", "theta_actor_v1_stairs_reach_from_trot_mid"),
        ("stairs_single", "trot_solid_fast", "stability", "theta_actor_v1_stairs_stability_from_trot_solid_fast"),

        ("tracer_sponge_firm_flat", "trot_soft_mid_clear", "balanced", "theta_actor_v1_sponge_balanced_from_trot_soft_mid_clear"),
        ("tracer_sponge_firm_flat", "trot_soft_mid_clear", "reach", "theta_actor_v1_sponge_reach_from_trot_soft_mid_clear"),
        ("tracer_sponge_firm_flat", "trot_soft_mid_clear", "stability", "theta_actor_v1_sponge_stability_from_trot_soft_mid_clear"),
    ]

    print("Generating v1 multimode residual actor actions:")
    for world, anchor, mode, new_name in specs:
        if anchor not in bank:
            raise KeyError(f"anchor not in bank: {anchor}")

        anchor_theta = {k: float(bank[anchor][k]) for k in THETA_KEYS}
        beta = WORLD_BETA[world]
        anchor_score = scores.get((world, anchor), 0.0)

        x = torch.tensor([feature_vec(world, mode, beta, anchor_theta, anchor_score)], dtype=torch.float32)

        with torch.no_grad():
            pred_norm = model(x)[0].tolist()

        delta = {
            k: float(pred_norm[i]) * DELTA_SCALE[k]
            for i, k in enumerate(THETA_KEYS)
        }

        actor_theta = {
            k: anchor_theta[k] + delta[k]
            for k in THETA_KEYS
        }
        actor_theta = project_theta(actor_theta)

        meta = {
            "world": world,
            "target_mode": mode,
            "anchor_action": anchor,
            "anchor_theta": anchor_theta,
            "pred_delta_norm": {k: pred_norm[i] for i, k in enumerate(THETA_KEYS)},
            "pred_delta": delta,
            "beta": {
                "beta_v": beta[0],
                "beta_s": beta[1],
                "beta_e": beta[2],
            },
            "anchor_score_task_beta": anchor_score,
        }

        add_action_to_world(data, world, new_name, actor_theta, meta)

        print()
        print(new_name)
        print("  world:", world)
        print("  mode:", mode)
        print("  anchor:", anchor)
        print("  anchor_score:", round(anchor_score, 4))
        print("  theta:", {k: round(actor_theta[k], 6) for k in THETA_KEYS})

    save_json(args.teacher_out, data)

    new_bank = build_action_bank(args.teacher_out)
    print()
    print("[wrote]", args.teacher_out)
    for _, _, _, name in specs:
        print(name, "readable:", name in new_bank, new_bank.get(name))


if __name__ == "__main__":
    main()
