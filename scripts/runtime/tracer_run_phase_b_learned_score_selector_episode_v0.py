#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, action_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.body(obs)
        return self.pi(h), self.v(h).squeeze(-1)


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def build_action_bank(teacher_json):
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


def load_policy(checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    worlds = ckpt["worlds"]
    actions = ckpt["actions"]
    world_to_i = ckpt["world_to_i"]

    model = PolicyValueNet(obs_dim=len(worlds), action_dim=len(actions), hidden=64)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, worlds, actions, world_to_i


def obs_from_world(world, world_to_i):
    if world not in world_to_i:
        raise RuntimeError(f"Unknown world for checkpoint: {world}. Available={sorted(world_to_i)}")
    obs = torch.zeros(len(world_to_i), dtype=torch.float32)
    obs[world_to_i[world]] = 1.0
    return obs


def raw_policy_distribution(model, actions, world_to_i, world, temperature):
    obs = obs_from_world(world, world_to_i).unsqueeze(0)

    with torch.no_grad():
        logits, value = model(obs)
        logits = logits[0] / max(float(temperature), 1e-6)
        probs = torch.softmax(logits, dim=-1)

    ranked = []
    for i, name in enumerate(actions):
        p = float(probs[i].item())
        ranked.append({
            "action_index": i,
            "action_name": name,
            "prob": p,
            "log_prob": math.log(max(p, 1e-12)),
            "raw_logit": float(logits[i].item()),
        })

    ranked = sorted(ranked, key=lambda x: x["prob"], reverse=True)

    return {
        "ranked_actions": ranked,
        "raw_argmax_action": ranked[0]["action_name"],
        "raw_argmax_prob": ranked[0]["prob"],
        "value": float(value[0].item()),
    }


def load_learned_scores(offline_score_json):
    report = load_json(offline_score_json)
    action_scores_by_world = {}

    for row in report.get("rows", []):
        world = row.get("world")
        policy = row.get("policy", "")

        # Use fixed baselines as action-level empirical learned scores.
        # Example: fixed_trot_mid -> trot_mid.
        if not policy.startswith("fixed_"):
            continue

        action = policy[len("fixed_"):]
        score = float(row.get("learned_objective_ram_score", 0.0))

        action_scores_by_world.setdefault(world, {})[action] = {
            "score": score,
            "policy": policy,
            "reach_rate": row.get("reach_rate"),
            "mean_R_v": row.get("mean_R_v"),
            "mean_R_s": row.get("mean_R_s"),
            "mean_R_e": row.get("mean_R_e"),
            "mean_final_rel_dist": row.get("mean_final_rel_dist"),
            "mean_post_reach_drift": row.get("mean_post_reach_drift"),
            "learned_shadow_used": row.get("learned_shadow_used"),
        }

    return action_scores_by_world


def normalize_scores(score_map):
    vals = [float(v["score"]) for v in score_map.values()]
    if not vals:
        return {}

    mu = sum(vals) / len(vals)
    var = sum((x - mu) ** 2 for x in vals) / max(len(vals), 1)
    std = math.sqrt(max(var, 1e-8))

    out = {}
    for action, obj in score_map.items():
        z = (float(obj["score"]) - mu) / std
        out[action] = dict(obj)
        out[action]["score_mean"] = mu
        out[action]["score_std"] = std
        out[action]["score_z"] = z

    return out


def select_with_learned_score(raw_dist, action_scores, alpha, missing_score_penalty):
    norm_scores = normalize_scores(action_scores)
    rescored = []

    for a in raw_dist["ranked_actions"]:
        name = a["action_name"]
        has_learned_score = name in norm_scores

        if has_learned_score:
            learned_z = float(norm_scores[name]["score_z"])
            learned_raw = float(norm_scores[name]["score"])
            learned_info = norm_scores[name]
            missing_penalty = 0.0
        else:
            learned_z = 0.0
            learned_raw = None
            learned_info = None
            missing_penalty = float(missing_score_penalty)

        final_score = float(a["log_prob"]) + float(alpha) * learned_z - missing_penalty

        item = dict(a)
        item.update({
            "has_learned_score": has_learned_score,
            "learned_score_raw": learned_raw,
            "learned_score_z": learned_z,
            "missing_score_penalty": missing_penalty,
            "final_score": final_score,
            "learned_score_info": learned_info,
        })
        rescored.append(item)

    rescored = sorted(rescored, key=lambda x: x["final_score"], reverse=True)
    chosen = rescored[0]

    return {
        "schema": "phase_b_learned_score_selector_output_v0",
        "alpha": alpha,
        "missing_score_penalty": missing_score_penalty,
        "raw_argmax_action": raw_dist["raw_argmax_action"],
        "raw_argmax_prob": raw_dist["raw_argmax_prob"],
        "selected_action": chosen["action_name"],
        "selected_action_prob_raw": chosen["prob"],
        "changed_action": chosen["action_name"] != raw_dist["raw_argmax_action"],
        "ranked_actions": rescored,
    }


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
    ap.add_argument("--checkpoint", default="artifacts/phase_b_ppo_theta_lite_online_update_v1_tracer_proxy/policy_value.pt")
    ap.add_argument("--teacher-json", default="configs/phase_b_ppo_warmstart_v0/ppo_warmstart_teacher_table.json")
    ap.add_argument("--offline-score-json", default="reports/phase_b_learned_objective_ram_offline_policy_score_v0.json")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--missing-score-penalty", type=float, default=0.15)
    ap.add_argument("--record-duration", type=float, default=35.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--runner-timeout", type=float, default=180.0)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    action_bank = build_action_bank(args.teacher_json)
    model, worlds, actions, world_to_i = load_policy(args.checkpoint)

    raw_dist = raw_policy_distribution(
        model=model,
        actions=actions,
        world_to_i=world_to_i,
        world=args.world_name,
        temperature=args.temperature,
    )

    scores_by_world = load_learned_scores(args.offline_score_json)
    world_scores = scores_by_world.get(args.world_name, {})

    selector = select_with_learned_score(
        raw_dist=raw_dist,
        action_scores=world_scores,
        alpha=args.alpha,
        missing_score_penalty=args.missing_score_penalty,
    )

    action_name = selector["selected_action"]
    if action_name not in action_bank:
        raise RuntimeError(f"Selected action {action_name} not in action bank. Available={sorted(action_bank)}")

    start = {
        "schema": "phase_b_learned_score_selector_episode_start_v0",
        "world_name": args.world_name,
        "checkpoint": args.checkpoint,
        "teacher_json": args.teacher_json,
        "offline_score_json": args.offline_score_json,
        "alpha": args.alpha,
        "missing_score_penalty": args.missing_score_penalty,
        "raw_policy_distribution": raw_dist,
        "learned_action_scores_for_world": world_scores,
        "learned_score_selector": selector,
        "selected_action_name": action_name,
        "theta_action": action_bank[action_name],
        "run_dir": str(out_root),
        "control_note": "Weak learned Objective/RAM intervention: raw PPO log-prob plus alpha times normalized offline learned score."
    }

    save_json(out_root / "learned_score_selector_episode_start_v0.json", start)
    print(json.dumps(start, indent=2, sort_keys=True))

    cmd = [
        "python3",
        "scripts/runtime/tracer_run_phase_b_fixed_theta_episode_v0.py",
        "--world-name", args.world_name,
        "--action-name", action_name,
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
            "schema": "phase_b_learned_score_selector_policy_episode_result_v0",
            "world_name": args.world_name,
            "selected_action_name": action_name,
            "status": "failed",
            "error": output[-4000:],
        }

    result["schema"] = "phase_b_learned_score_selector_policy_episode_result_v0"
    result["raw_policy_distribution"] = raw_dist
    result["learned_score_selector"] = selector
    result["selected_action_name"] = action_name
    result["theta_action"] = action_bank[action_name]
    result["alpha"] = args.alpha
    result["offline_score_json"] = args.offline_score_json

    save_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))

    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
