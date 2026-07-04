#!/usr/bin/env python3
import argparse
import json
import math
import os
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
    ckpt = torch.load(checkpoint, map_location="cpu")
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


def raw_policy_selection(model, actions, world_to_i, world, mode, temperature):
    obs = obs_from_world(world, world_to_i).unsqueeze(0)

    with torch.no_grad():
        logits, value = model(obs)
        logits = logits[0] / max(float(temperature), 1e-6)
        probs = torch.softmax(logits, dim=-1)

    if mode == "argmax":
        idx = int(torch.argmax(probs).item())
    elif mode == "sample":
        idx = int(torch.distributions.Categorical(probs=probs).sample().item())
    else:
        raise ValueError(mode)

    ranked = sorted(
        [
            {
                "action_index": i,
                "action_name": actions[i],
                "prob": float(probs[i].item()),
                "raw_logit": float(logits[i].item())
            }
            for i in range(len(actions))
        ],
        key=lambda x: x["prob"],
        reverse=True,
    )

    return {
        "action_index": idx,
        "action_name": actions[idx],
        "prob": float(probs[idx].item()),
        "ranked_actions": ranked,
        "value": float(value[0].item()),
    }


def apply_objective_selector(raw_selection, world, selector_cfg):
    world_cfg = selector_cfg.get("worlds", {}).get(world, {})
    bias = world_cfg.get("action_logit_bias", {}) or {}
    beta = world_cfg.get("beta", {})
    ram_proxy = world_cfg.get("ram_proxy", {})
    note = world_cfg.get("selection_note", "")

    ranked = raw_selection.get("ranked_actions", [])
    rescored = []

    for a in ranked:
        name = a["action_name"]
        p = max(float(a.get("prob", 0.0)), 1e-12)
        logit = math.log(p)
        b = float(bias.get(name, 0.0))
        rescored.append({
            "action_index": a.get("action_index"),
            "action_name": name,
            "original_prob": p,
            "bias": b,
            "biased_logit": logit + b,
        })

    if not rescored:
        raise RuntimeError("empty raw selection")

    max_logit = max(x["biased_logit"] for x in rescored)
    z = sum(math.exp(x["biased_logit"] - max_logit) for x in rescored)

    for x in rescored:
        x["prob"] = math.exp(x["biased_logit"] - max_logit) / max(z, 1e-12)

    rescored = sorted(rescored, key=lambda x: x["prob"], reverse=True)
    chosen = rescored[0]

    selection = {
        "action_index": chosen.get("action_index"),
        "action_name": chosen["action_name"],
        "prob": chosen["prob"],
        "ranked_actions": [
            {
                "action_index": x.get("action_index"),
                "action_name": x["action_name"],
                "prob": x["prob"],
                "original_prob": x["original_prob"],
                "bias": x["bias"]
            }
            for x in rescored
        ],
        "value": raw_selection.get("value"),
    }

    selector_info = {
        "schema": "phase_b_objective_selector_output_v0",
        "world_name": world,
        "terrain_class": world_cfg.get("terrain_class", "unknown"),
        "beta": beta,
        "ram_proxy": ram_proxy,
        "action_logit_bias": bias,
        "selection_note": note,
        "raw_action": raw_selection.get("action_name"),
        "selected_action": selection.get("action_name"),
        "changed_action": raw_selection.get("action_name") != selection.get("action_name"),
    }

    return selection, selector_info


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
    ap.add_argument("--selector-json", default="configs/phase_b_objective_selector_v0/objective_selector_ram_proxy_v0.json")
    ap.add_argument("--mode", choices=["argmax", "sample"], default="argmax")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--record-duration", type=float, default=35.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--runner-timeout", type=float, default=180.0)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    action_bank = build_action_bank(args.teacher_json)
    model, worlds, actions, world_to_i = load_policy(args.checkpoint)
    selector_cfg = load_json(args.selector_json)

    raw_selection = raw_policy_selection(
        model=model,
        actions=actions,
        world_to_i=world_to_i,
        world=args.world_name,
        mode=args.mode,
        temperature=args.temperature,
    )
    selection, selector_info = apply_objective_selector(raw_selection, args.world_name, selector_cfg)

    action_name = selection["action_name"]
    if action_name not in action_bank:
        raise RuntimeError(f"Selected action {action_name} not in action bank. Available={sorted(action_bank)}")

    start = {
        "schema": "phase_b_objective_selector_episode_start_v0",
        "world_name": args.world_name,
        "checkpoint": args.checkpoint,
        "teacher_json": args.teacher_json,
        "selector_json": args.selector_json,
        "mode": args.mode,
        "temperature": args.temperature,
        "raw_selection": raw_selection,
        "objective_selector": selector_info,
        "selection": selection,
        "selected_action_name": action_name,
        "theta_action": action_bank[action_name],
        "run_dir": str(out_root),
    }

    print(json.dumps(start, indent=2, sort_keys=True))
    save_json(out_root / "objective_selector_episode_start_v0.json", start)

    cmd = [
        "python3",
        "scripts/runtime/tracer_run_phase_b_fixed_theta_episode_v0.py",
        "--world-name",
        args.world_name,
        "--action-name",
        action_name,
        "--out-root",
        str(out_root),
        "--teacher-json",
        args.teacher_json,
        "--record-duration",
        str(args.record_duration),
        "--sample-hz",
        str(args.sample_hz),
        "--runner-timeout",
        str(args.runner_timeout),
    ]

    rc, output = run_cmd(cmd, timeout=args.runner_timeout + 30.0)

    result_path = out_root / "ppo_policy_episode_result_v0.json"
    if result_path.exists():
        result = load_json(result_path)
    else:
        result = {
            "schema": "phase_b_objective_selector_policy_episode_result_v0",
            "world_name": args.world_name,
            "selected_action_name": action_name,
            "status": "failed",
            "error": output[-4000:],
        }

    result["schema"] = "phase_b_objective_selector_policy_episode_result_v0"
    result["raw_selection"] = raw_selection
    result["objective_selector"] = selector_info
    result["selection"] = selection
    result["selected_action_name"] = action_name
    result["theta_action"] = action_bank[action_name]

    save_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))

    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
