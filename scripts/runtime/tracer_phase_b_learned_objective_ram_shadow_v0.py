#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn as nn


OBJECTIVE_CONTEXT_DEFAULTS = {
    "earth": {
        "slope_deg": 0.0,
        "slope_abs": 0.0,
        "downhill": 0.0,
        "uphill": 0.0,
        "terrain_backstep_context": 0.0,
        "mu": 0.90,
        "slip": 0.00,
        "roughness": 0.00,
        "stiffness_log": 12.0,
        "damping_log": 2.0,
        "soft_contact": 0.0,
        "softness": 0.0,
        "terrain_slippery": 0.0,
        "terrain_sponge": 0.0,
        "terrain_rough": 0.0,
        "terrain_flat": 1.0,
        "named_downslope_world": 0.0,
        "negvx_on_slope_world": 0.0,
        "postfix_flag": 0.0,
        "candidate_repeat_flag": 0.0,
        "stairs": 0.0,
        "sponge": 0.0,
        "flat": 1.0,
        "solid": 1.0,
        "risk_prior": 0.10,
        "hold_risk_prior": 0.10,
        "reach_risk_prior": 0.10,
        "uncertainty_prior": 0.10
    },
    "stairs_single": {
        "slope_deg": 0.0,
        "slope_abs": 0.0,
        "downhill": 0.0,
        "uphill": 0.0,
        "terrain_backstep_context": 0.0,
        "mu": 0.85,
        "slip": 0.02,
        "roughness": 0.45,
        "stiffness_log": 12.0,
        "damping_log": 2.0,
        "soft_contact": 0.0,
        "softness": 0.0,
        "terrain_slippery": 0.0,
        "terrain_sponge": 0.0,
        "terrain_rough": 1.0,
        "terrain_flat": 0.0,
        "named_downslope_world": 0.0,
        "negvx_on_slope_world": 0.0,
        "postfix_flag": 0.0,
        "candidate_repeat_flag": 0.0,
        "stairs": 1.0,
        "sponge": 0.0,
        "flat": 0.0,
        "solid": 1.0,
        "risk_prior": 0.35,
        "hold_risk_prior": 0.35,
        "reach_risk_prior": 0.30,
        "uncertainty_prior": 0.35
    },
    "tracer_sponge_firm_flat": {
        "slope_deg": 0.0,
        "slope_abs": 0.0,
        "downhill": 0.0,
        "uphill": 0.0,
        "terrain_backstep_context": 0.0,
        "mu": 0.75,
        "slip": 0.03,
        "roughness": 0.05,
        "stiffness_log": 8.0,
        "damping_log": 2.5,
        "soft_contact": 1.0,
        "softness": 1.0,
        "terrain_slippery": 0.0,
        "terrain_sponge": 1.0,
        "terrain_rough": 0.0,
        "terrain_flat": 1.0,
        "named_downslope_world": 0.0,
        "negvx_on_slope_world": 0.0,
        "postfix_flag": 0.0,
        "candidate_repeat_flag": 0.0,
        "stairs": 0.0,
        "sponge": 1.0,
        "flat": 1.0,
        "solid": 0.0,
        "risk_prior": 0.70,
        "hold_risk_prior": 0.90,
        "reach_risk_prior": 0.45,
        "uncertainty_prior": 0.70
    }
}


class ObjectiveIRLNet(nn.Module):
    def __init__(self, input_dim=19):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
        )
        self.beta_head = nn.Linear(48, 3)
        self.recovery_head = nn.Linear(48, 1)

    def forward(self, x):
        h = self.trunk(x)
        beta_logits = self.beta_head(h)
        beta = torch.softmax(beta_logits, dim=-1)
        recovery = torch.sigmoid(self.recovery_head(h)).squeeze(-1)
        return beta, recovery


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def maybe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def load_teacher_action_bank(teacher_json):
    t = load_json(teacher_json)
    bank = {}

    for _, obj in t.get("worlds", {}).items():
        for key in ["ranked_actions", "ppo_action_prior", "actions"]:
            for a in obj.get(key, []) or []:
                name = a.get("action_name") or a.get("name")
                theta = a.get("theta_action") or a.get("theta")
                if name and theta:
                    theta = dict(theta)
                    theta.setdefault("name", name)
                    bank[name] = theta

    if not bank:
        raise RuntimeError(f"No action bank found in {teacher_json}")

    return bank


def load_objective_irl_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    context_cols = ckpt["context_cols"]
    model = ObjectiveIRLNet(input_dim=len(context_cols))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, context_cols


def make_context_vector(world_name, context_cols):
    defaults = OBJECTIVE_CONTEXT_DEFAULTS.get(world_name, OBJECTIVE_CONTEXT_DEFAULTS["earth"])
    x = []
    missing = []

    for c in context_cols:
        if c in defaults:
            x.append(float(defaults[c]))
        else:
            # Unknown future columns default to 0 but are recorded.
            x.append(0.0)
            missing.append(c)

    return torch.tensor(x, dtype=torch.float32), missing


def query_objective_irl(world_name, model_path):
    model, cols = load_objective_irl_model(model_path)
    x, missing = make_context_vector(world_name, cols)

    with torch.no_grad():
        beta, recovery = model(x.unsqueeze(0))

    b = beta[0].detach().cpu().tolist()

    return {
        "model_path": model_path,
        "context_cols": cols,
        "missing_context_cols_defaulted_to_zero": missing,
        "beta": {
            "beta_velocity": float(b[0]),
            "beta_stability": float(b[1]),
            "beta_energy": float(b[2])
        },
        "recovery_score": float(recovery[0].item())
    }


def match_registry_group(world_name, theta, registry):
    groups = registry.get("groups", {})
    best = None
    best_dist = 1e18

    for key, obj in groups.items():
        k = obj.get("key", {})
        if k.get("world_name") != world_name:
            continue

        dist = 0.0
        for field in ["vx_far", "vx_near", "body_height", "swing_clearance", "goal_slow_distance"]:
            if field in theta and field in k:
                dist += abs(float(theta[field]) - float(k[field]))

        if dist < best_dist:
            best_dist = dist
            best = obj

    if best is None:
        defaults = OBJECTIVE_CONTEXT_DEFAULTS.get(world_name, OBJECTIVE_CONTEXT_DEFAULTS["earth"])
        return {
            "matched": False,
            "reason": "no_world_match",
            "fallback": "world_prior",
            "avg_future_risk": float(defaults.get("risk_prior", 0.5)),
            "future_uncertainty": float(defaults.get("uncertainty_prior", 0.75)),
            "high_variability": True,
            "majority_semantic": f"unknown_ram_registry_for_{world_name}",
            "avg_final_rel_dist": None,
            "std_like_final_rel_dist": None,
            "avg_odom_x_delta": None,
            "std_like_odom_x_delta": None
        }

    return {
        "matched": True,
        "match_distance": best_dist,
        "key_string": best.get("key_string"),
        "majority_semantic": best.get("majority_semantic"),
        "n": best.get("n"),
        "avg_future_risk": best.get("avg_future_risk"),
        "future_uncertainty": best.get("future_uncertainty"),
        "semantic_uncertainty": best.get("semantic_uncertainty"),
        "high_variability": best.get("high_variability"),
        "avg_final_rel_dist": best.get("avg_final_rel_dist"),
        "std_like_final_rel_dist": best.get("std_like_final_rel_dist"),
        "avg_odom_x_delta": best.get("avg_odom_x_delta"),
        "std_like_odom_x_delta": best.get("std_like_odom_x_delta")
    }


def read_policy_result(path):
    if not path or not Path(path).exists():
        return {}
    return load_json(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-name", required=True)
    ap.add_argument("--action-name", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--teacher-json", default="configs/phase_b_ppo_warmstart_v0/ppo_warmstart_teacher_table.json")
    ap.add_argument("--objective-model", default="artifacts/tracer_objective_irl_v1/model.pt")
    ap.add_argument("--ram-registry", default="configs/phase_b_objective_ram_uncertainty_v0/current_registry.json")
    ap.add_argument("--policy-result-json", default="")
    args = ap.parse_args()

    action_bank = load_teacher_action_bank(args.teacher_json)
    if args.action_name not in action_bank:
        raise RuntimeError(f"Unknown action {args.action_name}. Available={sorted(action_bank)}")

    theta = action_bank[args.action_name]

    objective = query_objective_irl(args.world_name, args.objective_model)
    registry = load_json(args.ram_registry)
    ram_match = match_registry_group(args.world_name, theta, registry)

    result = read_policy_result(args.policy_result_json)
    reward_components = result.get("reward_components", {})

    shadow = {
        "schema": "phase_b_learned_objective_ram_shadow_v0",
        "world_name": args.world_name,
        "action_name": args.action_name,
        "theta_action": theta,
        "objective_irl": objective,
        "ram_registry_match": ram_match,
        "policy_result_json": args.policy_result_json,
        "observed_reward_components": reward_components,
        "interpretation": {
            "mode": "shadow_only",
            "control_effect": "none",
            "note": "This adapter records learned Objective/RAM outputs without changing the selected action or low-level controller."
        }
    }

    save_json(Path(args.out_json), shadow)
    print(json.dumps(shadow, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
