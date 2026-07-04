#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def ff(x, default=0.0):
    try:
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

    worlds = t.get("worlds", {})
    for _, obj in worlds.items():
        for key in ["ranked_actions", "ppo_action_prior", "actions"]:
            for a in obj.get(key, []) or []:
                name = a.get("action_name") or a.get("name")
                theta = a.get("theta_action") or a.get("theta") or None
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


def reward_components(summary):
    reached = bool(summary.get("reached_stop_distance", False))
    min_dist = ff(summary.get("min_rel_dist"), 999.0)
    final_dist = ff(summary.get("final_rel_dist"), 999.0)
    dx = abs(ff(summary.get("odom_x_delta"), 0.0))
    progress = ff(summary.get("progress_initial_minus_min"), 0.0)
    yaw = abs(ff(summary.get("max_abs_mpc_yaw_rate"), 0.0))

    # Keep this compatible with previous Phase-B reach/hold style.
    reach_reward = 0.0
    reach_reward += 8.0 * progress
    reach_reward -= 1.5 * min_dist
    if reached:
        reach_reward += 3.0

    hold_reward = 0.0
    hold_reward += 2.0
    hold_reward -= 2.0 * final_dist
    hold_reward -= 0.5 * dx
    hold_reward -= 0.5 * yaw

    return {
        "reach_reward": reach_reward,
        "hold_reward": hold_reward,
        "reached_stop_distance": reached,
        "min_rel_dist": min_dist,
        "final_rel_dist": final_dist,
        "odom_x_delta_abs": dx,
        "max_abs_mpc_yaw_rate": yaw,
    }


def run_cmd(cmd, env=None, timeout=None):
    print("[run]", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        env=env,
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
    ap.add_argument("--action-name", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--teacher-json", default="configs/phase_b_ppo_warmstart_v0/ppo_warmstart_teacher_table.json")
    ap.add_argument("--record-duration", type=float, default=35.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--runner-timeout", type=float, default=180.0)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    bank = build_action_bank(args.teacher_json)
    if args.action_name not in bank:
        raise RuntimeError(f"Unknown action {args.action_name}. Available: {sorted(bank)}")

    theta = bank[args.action_name]

    start = {
        "schema": "phase_b_fixed_theta_episode_start_v0",
        "world_name": args.world_name,
        "selected_action_name": args.action_name,
        "theta_action": theta,
        "run_dir": str(out_root),
    }
    print(json.dumps(start, indent=2, sort_keys=True))
    save_json(out_root / "fixed_theta_episode_start_v0.json", start)

    env = dict(os.environ)
    env.update({
        "TRACER_PHASE_B_WORLD": args.world_name,
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

    rc, output = run_cmd(
        ["bash", "scripts/runtime/tracer_run_phase_b_goal_episode_v0.sh"],
        env=env,
        timeout=args.runner_timeout,
    )

    summary_path = out_root / "phase_b_summary.json"
    result = {
        "schema": "phase_b_fixed_theta_policy_episode_result_v0",
        "world_name": args.world_name,
        "selected_action_name": args.action_name,
        "theta_action": theta,
        "status": "ok" if rc == 0 and summary_path.exists() else "failed",
        "summary_json": str(summary_path) if summary_path.exists() else None,
        "error": "" if rc == 0 else output[-4000:],
        "selection": {
            "action_name": args.action_name,
            "action_index": None,
            "prob": 1.0,
            "ranked_actions": [
                {
                    "action_name": args.action_name,
                    "prob": 1.0
                }
            ],
            "value": None,
        },
        "metrics": {},
        "reward_components": {},
    }

    if summary_path.exists():
        s = load_json(summary_path)
        result["metrics"] = {
            "reached_stop_distance": bool(s.get("reached_stop_distance", False)),
            "min_rel_dist": ff(s.get("min_rel_dist"), 999.0),
            "final_rel_dist": ff(s.get("final_rel_dist"), 999.0),
            "progress_initial_minus_min": ff(s.get("progress_initial_minus_min"), 0.0),
            "odom_x_delta": ff(s.get("odom_x_delta"), 0.0),
            "max_abs_mpc_yaw_rate": ff(s.get("max_abs_mpc_yaw_rate"), 0.0),
        }
        result["reward_components"] = reward_components(s)

    # Use this filename so the existing reach-terminated report script can reuse it.
    save_json(out_root / "ppo_policy_episode_result_v0.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
