#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn


class UtilityNet(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RAMEpisodeNet(nn.Module):
    def __init__(self, input_dim, hidden=64, bool_dim=5, reg_dim=1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
        )
        self.bool_head = nn.Linear(32, bool_dim)
        self.risk_head = nn.Linear(32, reg_dim)

    def forward(self, x):
        h = self.trunk(x)
        return self.bool_head(h), torch.sigmoid(self.risk_head(h))


class CandidateScorer(nn.Module):
    def __init__(self, input_dim, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def ff(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def build_action_bank(teacher_json):
    t = load_json(teacher_json)
    bank = {}

    def add_action(name, theta):
        if not name or not theta:
            return
        theta = dict(theta)
        theta.setdefault("name", name)
        bank[name] = theta

    for _, obj in t.get("worlds", {}).items():
        for key in ["ranked_actions", "ppo_action_prior", "actions"]:
            for a in obj.get(key, []) or []:
                name = a.get("action_name") or a.get("name")
                theta = a.get("theta_action") or a.get("theta")
                if theta is None and name:
                    maybe = {
                        k: a[k]
                        for k in [
                            "vx_far",
                            "vx_near",
                            "goal_slow_distance",
                            "goal_stop_distance",
                            "body_height",
                            "swing_clearance",
                        ]
                        if k in a
                    }
                    theta = maybe if maybe else None
                add_action(name, theta)

        if "top_action" in obj:
            a = obj["top_action"]
            add_action(a.get("action_name") or a.get("name"), a.get("theta_action") or a.get("theta"))

    if not bank:
        raise RuntimeError(f"No actions found in teacher json: {teacher_json}")

    return bank


def world_features(world):
    return {
        "world_onehot:earth": 1.0 if world == "earth" else 0.0,
        "world_onehot:stairs_single": 1.0 if world == "stairs_single" else 0.0,
        "world_onehot:tracer_sponge_firm_flat": 1.0 if world == "tracer_sponge_firm_flat" else 0.0,
        "terrain_is_flat": 1.0 if world in ["earth", "tracer_sponge_firm_flat"] else 0.0,
        "terrain_is_stairs": 1.0 if world == "stairs_single" else 0.0,
        "terrain_is_soft_contact": 1.0 if world == "tracer_sponge_firm_flat" else 0.0,
        "terrain_roughness_proxy": 0.45 if world == "stairs_single" else 0.05 if world == "tracer_sponge_firm_flat" else 0.0,
        "terrain_slip_proxy": 0.03 if world == "tracer_sponge_firm_flat" else 0.02 if world == "stairs_single" else 0.0,
        "terrain_stiffness_log_proxy": 8.0 if world == "tracer_sponge_firm_flat" else 12.0,
    }


def candidate_features(world, theta):
    feat = {}
    feat.update(world_features(world))
    for k in [
        "vx_far",
        "vx_near",
        "goal_slow_distance",
        "goal_stop_distance",
        "body_height",
        "swing_clearance",
    ]:
        feat[f"theta:{k}"] = ff(theta.get(k))
    return feat


def vec(feat, names):
    return torch.tensor([float((feat or {}).get(k, 0.0)) for k in names], dtype=torch.float32)


def load_objective(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = UtilityNet(ckpt["input_dim"], hidden=ckpt.get("hidden_dim", 64))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt, model


def load_ram(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = RAMEpisodeNet(
        ckpt["input_dim"],
        hidden=ckpt.get("hidden_dim", 64),
        bool_dim=len(ckpt["bool_label_names"]),
        reg_dim=len(ckpt["reg_label_names"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt, model


def load_candidate_policy(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CandidateScorer(ckpt["input_dim"], hidden=96)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt, model


def score_candidates(world, action_bank, action_names, objective_path, ram_path, candidate_policy_path):
    obj_ckpt, obj = load_objective(objective_path)
    ram_ckpt, ram = load_ram(ram_path)
    pol_ckpt, pol = load_candidate_policy(candidate_policy_path)

    x_mean = torch.tensor(pol_ckpt["input_mean"], dtype=torch.float32)
    x_std = torch.tensor(pol_ckpt["input_std"], dtype=torch.float32)
    target_mean = float(pol_ckpt["target_mean"])
    target_std = float(pol_ckpt["target_std"])

    rows = []

    with torch.no_grad():
        for action_name in action_names:
            if action_name not in action_bank:
                continue

            theta = action_bank[action_name]
            feat = candidate_features(world, theta)

            obj_x = vec(feat, obj_ckpt["feature_names"]).unsqueeze(0)
            ram_x = vec(feat, ram_ckpt["feature_names"]).unsqueeze(0)

            objective_utility = float(obj(obj_x)[0].item())
            bool_logits, risk = ram(ram_x)
            bool_probs = torch.sigmoid(bool_logits[0]).detach().cpu().tolist()
            ram_risk = float(risk[0, 0].item())

            raw_vec = vec(feat, obj_ckpt["feature_names"]).tolist()
            raw_vec += [objective_utility, ram_risk]
            raw_vec += [float(x) for x in bool_probs]

            # v2 robust candidate policies include group-level robust target features.
            # v1 checkpoints do not have group_targets, so this block is skipped for v1.
            group_targets = pol_ckpt.get("group_targets", {})
            group_key = f"{world}::{action_name}"
            if group_key in group_targets:
                gt = group_targets[group_key]
                raw_vec += [
                    float(gt.get("mean_score", 0.0)),
                    float(gt.get("std_score", 0.0)),
                    float(gt.get("mean_risk", 0.0)),
                    float(gt.get("reach_rate", 0.0)),
                ]

            x = torch.tensor(raw_vec, dtype=torch.float32)

            if len(x) != len(x_mean):
                raise RuntimeError(
                    f"candidate feature dim mismatch for {world}::{action_name}: "
                    f"got {len(x)}, expected {len(x_mean)}; "
                    f"policy_schema={pol_ckpt.get('schema')}; "
                    f"input_names={pol_ckpt.get('input_names')}"
                )

            x_norm = (x - x_mean) / x_std

            score_z = float(pol(x_norm.unsqueeze(0))[0].item())
            score_raw = score_z * target_std + target_mean

            rows.append({
                "action_name": action_name,
                "theta_action": theta,
                "candidate_policy_score": score_raw,
                "candidate_policy_score_z": score_z,
                "objective_utility": objective_utility,
                "ram_risk": ram_risk,
                "ram_bool_probs": {
                    name: float(val)
                    for name, val in zip(ram_ckpt["bool_label_names"], bool_probs)
                },
                "features": feat,
                "group_target": group_targets.get(group_key),
                "policy_schema": pol_ckpt.get("schema"),
            })

    rows = sorted(rows, key=lambda r: r["candidate_policy_score"], reverse=True)
    return rows



def run_cmd(cmd, timeout=None):
    print("[run]", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    print(proc.stdout)
    return proc.returncode, proc.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-name", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--teacher-json", default="configs/phase_b_ppo_warmstart_v0/ppo_warmstart_teacher_table.json")
    ap.add_argument("--objective-model", default="artifacts/phase_b_objective_selector_pref_v2_clean/model.pt")
    ap.add_argument("--ram-model", default="artifacts/phase_b_ram_episode_v2_clean/model.pt")
    ap.add_argument("--candidate-policy", default="artifacts/phase_b_beta_ram_candidate_policy_v1/model.pt")
    ap.add_argument("--actions", default="trot_mid,trot_solid_fast,trot_cautious,trot_soft_mid_clear")
    ap.add_argument("--record-duration", type=float, default=35.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--runner-timeout", type=float, default=180.0)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    bank = build_action_bank(args.teacher_json)
    action_names = [a.strip() for a in args.actions.split(",") if a.strip()]

    ranked = score_candidates(
        world=args.world_name,
        action_bank=bank,
        action_names=action_names,
        objective_path=args.objective_model,
        ram_path=args.ram_model,
        candidate_policy_path=args.candidate_policy,
    )

    if not ranked:
        raise RuntimeError("No candidates scored.")

    selected = ranked[0]["action_name"]

    start = {
        "schema": "phase_b_beta_ram_candidate_policy_episode_start_v1",
        "world_name": args.world_name,
        "selected_action_name": selected,
        "theta_action": bank[selected],
        "ranked_candidates": ranked,
        "teacher_json": args.teacher_json,
        "objective_model": args.objective_model,
        "ram_model": args.ram_model,
        "candidate_policy": args.candidate_policy,
        "run_dir": str(out_root),
    }

    save_json(out_root / "beta_ram_candidate_policy_episode_start_v1.json", start)
    print(json.dumps(start, indent=2, sort_keys=True))

    cmd = [
        "python3",
        "scripts/runtime/tracer_run_phase_b_fixed_theta_episode_v0.py",
        "--world-name", args.world_name,
        "--action-name", selected,
        "--out-root", str(out_root),
        "--teacher-json", args.teacher_json,
        "--record-duration", str(args.record_duration),
        "--sample-hz", str(args.sample_hz),
        "--runner-timeout", str(args.runner_timeout),
    ]

    rc, output = run_cmd(cmd, timeout=args.runner_timeout + 30.0)

    result_path = out_root / "ppo_policy_episode_result_v0.json"
    if result_path.exists():
        result = load_json(result_path)
    else:
        result = {
            "schema": "phase_b_beta_ram_candidate_policy_episode_result_v1",
            "world_name": args.world_name,
            "selected_action_name": selected,
            "status": "failed",
            "error": output[-4000:],
        }

    result["schema"] = "phase_b_beta_ram_candidate_policy_episode_result_v1"
    result["beta_ram_candidate_policy"] = {
        "selected_action_name": selected,
        "ranked_candidates": ranked,
    }

    save_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))

    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
