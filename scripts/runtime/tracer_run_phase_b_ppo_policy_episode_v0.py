#!/usr/bin/env python3
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


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


def obs_from_world(world, world_to_i):
    x = torch.zeros(len(world_to_i), dtype=torch.float32)
    if world not in world_to_i:
        raise KeyError(f"world {world!r} not in PPO checkpoint worlds={list(world_to_i)}")
    x[world_to_i[world]] = 1.0
    return x


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    worlds = ckpt["worlds"]
    actions = ckpt["actions"]
    world_to_i = ckpt["world_to_i"]
    action_to_i = ckpt["action_to_i"]

    model = PolicyValueNet(obs_dim=len(worlds), action_dim=len(actions), hidden=64)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, worlds, actions, world_to_i, action_to_i, ckpt


def choose_action(model, world, actions, world_to_i, mode, temperature):
    obs = obs_from_world(world, world_to_i).unsqueeze(0)
    with torch.no_grad():
        logits, value = model(obs)
        logits = logits[0] / max(float(temperature), 1e-6)
        probs = torch.softmax(logits, dim=-1)

        if mode == "argmax":
            idx = int(torch.argmax(probs).item())
        elif mode == "sample":
            dist = torch.distributions.Categorical(probs=probs)
            idx = int(dist.sample().item())
        else:
            raise ValueError(f"unknown mode={mode}")

        ranked = sorted(
            [
                {
                    "action_name": actions[i],
                    "prob": float(probs[i].item()),
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
        "value": float(value[0].item()),
        "ranked_actions": ranked,
    }


def build_action_bank_from_teacher(teacher_json):
    teacher = load_json(teacher_json)
    bank = {}

    for world, obj in teacher["worlds"].items():
        for stat in obj.get("ranked_actions", []):
            name = stat["action_name"]
            theta = stat.get("theta_action") or {}
            if name and theta:
                bank[name] = theta

    return bank


def run_cmd(cmd, timeout=None, env=None):
    print("[run]", " ".join(cmd), flush=True)
    if env:
        shown = {k: env[k] for k in sorted(env) if k.startswith("TRACER_PHASE_B_")}
        print("[env]", json.dumps(shown, indent=2, sort_keys=True), flush=True)
    p = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=env,
    )
    print(p.stdout)
    return p.returncode, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-name", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--checkpoint", default="artifacts/phase_b_ppo_theta_lite_discrete_v0/policy_value.pt")
    ap.add_argument("--teacher-json", default="configs/phase_b_ppo_warmstart_v0/ppo_warmstart_teacher_table.json")
    ap.add_argument("--mode", choices=["argmax", "sample"], default="sample")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--record-duration", type=float, default=35.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--runner-timeout", type=float, default=180.0)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model, worlds, actions, world_to_i, action_to_i, ckpt = load_model(args.checkpoint)
    action_bank = build_action_bank_from_teacher(args.teacher_json)

    chosen = choose_action(
        model=model,
        world=args.world_name,
        actions=actions,
        world_to_i=world_to_i,
        mode=args.mode,
        temperature=args.temperature,
    )

    action_name = chosen["action_name"]
    if action_name not in action_bank:
        raise KeyError(f"action {action_name!r} missing from teacher action bank")

    theta = dict(action_bank[action_name])
    theta["name"] = action_name

    start_obj = {
        "schema": "phase_b_ppo_policy_episode_start_v0",
        "world_name": args.world_name,
        "selected_action_name": action_name,
        "selection": chosen,
        "theta_action": theta,
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "temperature": args.temperature,
        "run_dir": str(out_root),
    }
    save_json(out_root / "ppo_policy_episode_start_v0.json", start_obj)
    print(json.dumps(start_obj, indent=2, sort_keys=True))

    cmd = [
        "bash",
        "scripts/runtime/tracer_run_phase_b_goal_episode_v0.sh",
    ]

    env = dict(__import__("os").environ)
    env.update({
        "TRACER_PHASE_B_WORLD": str(args.world_name),
        "TRACER_PHASE_B_RUN_DIR": str(out_root),
        "TRACER_PHASE_B_RECORD_DURATION": str(args.record_duration),
        "TRACER_PHASE_B_SAMPLE_HZ": str(args.sample_hz),
        "TRACER_PHASE_B_GOAL_DISTANCE_AHEAD": "0.5",
        "TRACER_PHASE_B_VX_FAR": str(theta["vx_far"]),
        "TRACER_PHASE_B_VX_NEAR": str(theta["vx_near"]),
        "TRACER_PHASE_B_GOAL_SLOW_DISTANCE": str(theta["goal_slow_distance"]),
        "TRACER_PHASE_B_GOAL_STOP_DISTANCE": str(theta["goal_stop_distance"]),
        "TRACER_PHASE_B_BODY_HEIGHT": str(theta["body_height"]),
        "TRACER_PHASE_B_SWING_CLEARANCE": str(theta["swing_clearance"]),
    })

    status = "ok"
    error = ""
    try:
        rc, output = run_cmd(cmd, timeout=args.runner_timeout, env=env)
        if rc != 0:
            status = "runner_nonzero"
            error = f"returncode={rc}"
    except subprocess.TimeoutExpired as e:
        status = "timeout"
        error = str(e)

    summary_path = out_root / "phase_b_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}

    # Reuse simple reach metrics from summary.
    reached = bool(summary.get("reached_stop_distance", False))
    min_rel_dist = float(summary.get("min_rel_dist", 999.0))
    final_rel_dist = float(summary.get("final_rel_dist", 999.0))
    progress = float(summary.get("progress_initial_minus_min", 0.0))
    odom_x_delta = float(summary.get("odom_x_delta", 0.0))
    max_abs_yaw = float(summary.get("max_abs_mpc_yaw_rate", 0.0))

    reach_reward = 8.0 * max(0.0, 1.0 - min_rel_dist / 0.50)
    if reached:
        reach_reward += 1.0
    reach_reward += 0.5 * progress
    reach_reward -= 0.25 * max_abs_yaw

    hold_reward = 2.0 - 2.0 * final_rel_dist - 0.5 * abs(odom_x_delta)

    result = {
        "schema": "phase_b_ppo_policy_episode_result_v0",
        "status": status,
        "error": error,
        "world_name": args.world_name,
        "selected_action_name": action_name,
        "theta_action": theta,
        "selection": chosen,
        "summary_json": str(summary_path) if summary_path.exists() else None,
        "metrics": {
            "reached_stop_distance": reached,
            "min_rel_dist": min_rel_dist,
            "final_rel_dist": final_rel_dist,
            "progress_initial_minus_min": progress,
            "odom_x_delta": odom_x_delta,
            "max_abs_mpc_yaw_rate": max_abs_yaw,
        },
        "reward_components": {
            "reach_reward": reach_reward,
            "hold_reward": hold_reward,
            "reached_stop_distance": reached,
            "min_rel_dist": min_rel_dist,
            "final_rel_dist": final_rel_dist,
            "odom_x_delta_abs": abs(odom_x_delta),
            "max_abs_mpc_yaw_rate": max_abs_yaw,
        },
    }

    save_json(out_root / "ppo_policy_episode_result_v0.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
