#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path


def ff(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def classify(summary):
    reached = bool(summary.get("reached_stop_distance", False))
    min_dist = ff(summary.get("min_rel_dist"), 999.0)
    final_dist = ff(summary.get("final_rel_dist"), 999.0)
    dx = abs(ff(summary.get("odom_x_delta"), 0.0))
    progress = ff(summary.get("progress_initial_minus_min"), 0.0)

    if reached and final_dist <= 0.30 and dx <= 0.80:
        return "stable_goal_reach_flat_locomotion"
    if reached and (final_dist > 0.75 or dx > 1.0):
        return "reached_but_failed_to_hold"
    if final_dist > 0.75 or dx > 1.0:
        return "forward_walk_unreliable_on_soft_terrain"
    if reached or min_dist <= 0.18:
        return "approach_possible_but_post_reach_hold_needed"
    if progress > 0.08:
        return "cautious_probe_required"
    return "no_meaningful_progress"


def reward(summary, semantic):
    final_dist = ff(summary.get("final_rel_dist"), 999.0)
    dx = abs(ff(summary.get("odom_x_delta"), 0.0))
    yaw = abs(ff(summary.get("max_abs_mpc_yaw_rate"), 0.0))
    progress = ff(summary.get("progress_initial_minus_min"), 0.0)
    reached = bool(summary.get("reached_stop_distance", False))

    r = 0.0
    r += 2.0 * max(progress, 0.0)
    r -= min(final_dist, 5.0)
    r -= 0.5 * min(dx, 5.0)
    r -= 0.5 if yaw >= 0.299 else 0.0

    if reached:
        r += 1.0
    if semantic == "stable_goal_reach_flat_locomotion":
        r += 4.0
    elif semantic == "approach_possible_but_post_reach_hold_needed":
        r += 0.5
    elif semantic == "reached_but_failed_to_hold":
        # Reaching the goal is useful information, but large post-reach drift is
        # unsafe and should be penalized. This separates gait progress from
        # missing post-reach hold/latch behavior.
        r -= 2.0
    elif semantic == "forward_walk_unreliable_on_soft_terrain":
        r -= 4.0
    elif semantic == "no_meaningful_progress":
        r -= 1.0

    return r


def find_summary(run_dir):
    run_dir = Path(run_dir)
    candidates = [
        run_dir / "episode" / "phase_b_summary.json",
        run_dir / "phase_b_summary.json",
    ]
    candidates.extend(sorted(run_dir.glob("**/phase_b_summary.json")))
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_json(path, default):
    p = Path(path)
    if p.is_file():
        return json.load(open(p))
    return default


def save_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def init_policy(action_bank, context_registry):
    actions = action_bank["actions"]
    worlds = sorted(context_registry["worlds"].keys())
    return {
        "schema": "phase_b_theta_lite_bandit_policy_v0",
        "created_at": time.strftime("%Y%m%d_%H%M%S"),
        "action_bank_schema": action_bank.get("schema"),
        "context_registry_schema": context_registry.get("schema"),
        "worlds": {
            w: {
                "n_total": 0,
                "actions": {
                    a["name"]: {
                        "n": 0,
                        "mean_reward": 0.0,
                        "total_reward": 0.0,
                        "best_reward": None,
                        "last_reward": None,
                        "semantic_counts": {},
                    }
                    for a in actions
                }
            }
            for w in worlds
        }
    }


def select_action(policy, action_bank, world, epsilon, ucb_c):
    actions = action_bank["actions"]
    world_stats = policy["worlds"].setdefault(world, {
        "n_total": 0,
        "actions": {}
    })

    for a in actions:
        world_stats["actions"].setdefault(a["name"], {
            "n": 0,
            "mean_reward": 0.0,
            "total_reward": 0.0,
            "best_reward": None,
            "last_reward": None,
            "semantic_counts": {},
        })

    # Try unvisited actions first.
    unvisited = [
        a for a in actions
        if world_stats["actions"][a["name"]]["n"] == 0
    ]
    if unvisited:
        return unvisited[0], "unvisited"

    if random.random() < epsilon:
        return random.choice(actions), "epsilon_random"

    n_total = max(1, int(world_stats.get("n_total", 0)))
    scored = []
    for a in actions:
        st = world_stats["actions"][a["name"]]
        n = max(1, int(st["n"]))
        mean = float(st["mean_reward"])
        ucb = mean + ucb_c * math.sqrt(math.log(n_total + 1.0) / n)
        scored.append((ucb, mean, a))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][2], "ucb"


def update_policy(policy, world, action_name, rew, sem):
    world_stats = policy["worlds"].setdefault(world, {"n_total": 0, "actions": {}})
    st = world_stats["actions"].setdefault(action_name, {
        "n": 0,
        "mean_reward": 0.0,
        "total_reward": 0.0,
        "best_reward": None,
        "last_reward": None,
        "semantic_counts": {},
    })

    st["n"] += 1
    st["total_reward"] += rew
    st["mean_reward"] = st["total_reward"] / st["n"]
    st["last_reward"] = rew
    st["best_reward"] = rew if st["best_reward"] is None else max(st["best_reward"], rew)
    st["semantic_counts"][sem] = st["semantic_counts"].get(sem, 0) + 1
    world_stats["n_total"] = int(world_stats.get("n_total", 0)) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-name", required=True)
    ap.add_argument("--policy-json", default="configs/phase_b_theta_lite_rl_v0/current_bandit_policy.json")
    ap.add_argument("--action-bank-json", default="configs/phase_b_theta_lite_rl_v0/action_bank_v0.json")
    ap.add_argument("--context-registry-json", default="configs/phase_b_theta_lite_rl_v0/context_registry_v0.json")
    ap.add_argument("--runner", default="scripts/runtime/tracer_run_phase_b_goal_episode_v0.sh")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--record-duration", type=float, default=35.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--ucb-c", type=float, default=1.0)
    ap.add_argument("--runner-timeout", type=float, default=140.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.seed:
        random.seed(args.seed + int(time.time()))

    action_bank = load_json(args.action_bank_json, None)
    context_registry = load_json(args.context_registry_json, None)
    if action_bank is None:
        raise FileNotFoundError(args.action_bank_json)
    if context_registry is None:
        raise FileNotFoundError(args.context_registry_json)

    policy = load_json(args.policy_json, None)
    if policy is None:
        policy = init_policy(action_bank, context_registry)

    action, select_reason = select_action(
        policy, action_bank, args.world_name, args.epsilon, args.ucb_c
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root or f"artifacts/phase_b_theta_lite_bandit_episode_v0_{stamp}_{args.world_name}_{action['name']}")
    out_root.mkdir(parents=True, exist_ok=True)

    context_info = context_registry["worlds"].get(args.world_name, {})
    decision = {
        "schema": "phase_b_theta_lite_bandit_decision_v0",
        "world_name": args.world_name,
        "terrain_label": context_info.get("terrain_label"),
        "context_fields": context_registry.get("context_fields"),
        "context": context_info.get("context"),
        "selected_action_name": action["name"],
        "selected_theta_lite": action,
        "selection_reason": select_reason,
        "epsilon": args.epsilon,
        "ucb_c": args.ucb_c,
        "policy_json": args.policy_json,
    }
    save_json(out_root / "bandit_decision_v0.json", decision)

    env = os.environ.copy()
    env.update({
        "TRACER_PHASE_B_WORLD": args.world_name,
        "TRACER_PHASE_B_RUN_DIR": str(out_root),
        "TRACER_PHASE_B_RECORD_DURATION": str(args.record_duration),
        "TRACER_PHASE_B_SAMPLE_HZ": str(args.sample_hz),

        "TRACER_PHASE_B_GOAL_DISTANCE_AHEAD": "0.5",
        "TRACER_PHASE_B_GOAL_STOP_DISTANCE": str(action["goal_stop_distance"]),
        "TRACER_PHASE_B_GOAL_SLOW_DISTANCE": str(action["goal_slow_distance"]),

        "TRACER_PHASE_B_VX_FAR": str(action["vx_far"]),
        "TRACER_PHASE_B_VX_NEAR": str(action["vx_near"]),
        "TRACER_PHASE_B_BODY_HEIGHT": str(action["body_height"]),
        "TRACER_PHASE_B_SWING_CLEARANCE": str(action["swing_clearance"]),

        "TRACER_PHASE_B_THETA_LITE_RL_ACTION_NAME": action["name"],
        "TRACER_PHASE_B_THETA_LITE_RL_DECISION_JSON": str(out_root / "bandit_decision_v0.json"),
    })

    print(json.dumps({
        "schema": "phase_b_theta_lite_bandit_episode_start_v0",
        "world_name": args.world_name,
        "run_dir": str(out_root),
        "selected_action": action,
        "selection_reason": select_reason,
    }, indent=2, sort_keys=True))

    status = "ok"
    error = ""
    try:
        subprocess.check_call(["bash", args.runner], env=env, timeout=args.runner_timeout)
    except subprocess.TimeoutExpired as e:
        status = "timeout"
        error = f"runner timeout after {args.runner_timeout}s: {e}"
    except subprocess.CalledProcessError as e:
        status = "failed"
        error = str(e)

    summary_path = find_summary(out_root)
    policy_updated = False

    if summary_path is None:
        sem = "runner_failed_no_summary"
        rew = -10.0
        summary = {}
        # Runtime failures should be recorded but should not poison the learned
        # gait/action policy. They are simulator/bridge failures, not terrain
        # locomotion outcomes.
        policy_updated = False
    else:
        summary = json.load(open(summary_path))
        sem = classify(summary)
        rew = reward(summary, sem)
        update_policy(policy, args.world_name, action["name"], rew, sem)
        save_json(args.policy_json, policy)
        policy_updated = True

    result = {
        "schema": "phase_b_theta_lite_bandit_episode_result_v0",
        "world_name": args.world_name,
        "run_dir": str(out_root),
        "status": status,
        "error": error,
        "selected_action_name": action["name"],
        "theta_action": action,
        "selection_reason": select_reason,
        "summary_json": str(summary_path) if summary_path else None,
        "semantic": sem,
        "reward": rew,
        "metrics": {
            "reached_stop_distance": summary.get("reached_stop_distance"),
            "min_rel_dist": summary.get("min_rel_dist"),
            "final_rel_dist": summary.get("final_rel_dist"),
            "progress_initial_minus_min": summary.get("progress_initial_minus_min"),
            "odom_x_delta": summary.get("odom_x_delta"),
            "max_abs_mpc_yaw_rate": summary.get("max_abs_mpc_yaw_rate"),
        },
        "policy_json": args.policy_json,
        "policy_updated": policy_updated,
    }

    save_json(out_root / "bandit_episode_result_v0.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
