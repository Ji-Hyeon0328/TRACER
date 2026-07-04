#!/usr/bin/env python3
import argparse
import json
import math
import random
import subprocess
from pathlib import Path

import torch

# Reuse v1 runner internals.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.runtime.tracer_run_phase_b_beta_ram_candidate_policy_episode_v1 import (
    build_action_bank,
    score_candidates,
)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def softmax_sample(rows, temperature, top_k, epsilon, rng):
    if not rows:
        raise RuntimeError("No candidate rows to sample from.")

    ranked = list(rows)
    if top_k > 0:
        ranked = ranked[:top_k]

    if epsilon > 0.0 and rng.random() < epsilon:
        selected = rng.choice(ranked)
        probs = {r["action_name"]: 1.0 / len(ranked) for r in ranked}
        return selected, probs, "epsilon_uniform"

    scores = [float(r["candidate_policy_score"]) for r in ranked]
    t = max(float(temperature), 1e-6)
    m = max(scores)
    exps = [math.exp((s - m) / t) for s in scores]
    z = sum(exps)
    probs_list = [e / z for e in exps]

    u = rng.random()
    acc = 0.0
    idx = len(ranked) - 1
    for i, p in enumerate(probs_list):
        acc += p
        if u <= acc:
            idx = i
            break

    probs = {
        r["action_name"]: float(p)
        for r, p in zip(ranked, probs_list)
    }
    return ranked[idx], probs, "softmax_topk"


def run_cmd(cmd, timeout=None):
    print("[run]", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    print(proc.stdout, flush=True)
    return proc.returncode, proc.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-name", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--teacher-json", default="configs/phase_b_training_pipeline_v2_sponge_expanded/ppo_warmstart_teacher_table_sponge_expanded_v0.json")
    ap.add_argument("--objective-model", default="artifacts/phase_b_objective_selector_pref_v3_merged_clean/model.pt")
    ap.add_argument("--ram-model", default="artifacts/phase_b_ram_episode_v3_merged_clean/model.pt")
    ap.add_argument("--candidate-policy", default="artifacts/phase_b_beta_ram_candidate_policy_v3_robust_clean/model.pt")
    ap.add_argument("--actions", default="trot_mid,trot_solid_fast,trot_cautious,trot_soft_mid_clear,sponge_slow_high_clear,sponge_short_step_stable,sponge_reach_then_brake,sponge_mid_brake_clear,sponge_probe_crawlish")
    ap.add_argument("--record-duration", type=float, default=35.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--runner-timeout", type=float, default=180.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--epsilon", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

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

    selected_row, probs, selection_mode = softmax_sample(
        ranked,
        temperature=args.temperature,
        top_k=args.top_k,
        epsilon=args.epsilon,
        rng=rng,
    )

    selected = selected_row["action_name"]

    start = {
        "schema": "phase_b_beta_ram_candidate_policy_episode_start_v2_sample",
        "world_name": args.world_name,
        "selected_action_name": selected,
        "theta_action": bank[selected],
        "selection_mode": selection_mode,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "selection_probs": probs,
        "ranked_candidates": ranked,
        "teacher_json": args.teacher_json,
        "objective_model": args.objective_model,
        "ram_model": args.ram_model,
        "candidate_policy": args.candidate_policy,
        "run_dir": str(out_root),
    }
    save_json(out_root / "beta_ram_candidate_policy_episode_start_v2_sample.json", start)
    print(json.dumps(start, indent=2, sort_keys=True), flush=True)

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
            "status": "failed",
            "world_name": args.world_name,
            "selected_action_name": selected,
            "error": output[-4000:],
        }

    result["schema"] = "phase_b_beta_ram_candidate_policy_episode_result_v2_sample"
    result["beta_ram_candidate_policy"] = {
        "selected_action_name": selected,
        "selection_mode": selection_mode,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "selection_probs": probs,
        "ranked_candidates": ranked,
    }

    save_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
