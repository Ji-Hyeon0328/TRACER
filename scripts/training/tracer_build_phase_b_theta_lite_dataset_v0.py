#!/usr/bin/env python3

import argparse
import glob
import json
import math
from collections import Counter
from pathlib import Path


def ff(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def bb(x):
    return bool(x)


def classify(summary):
    stable = bb(summary.get("stable_reached", False))
    reached = bb(summary.get("reached_stop_distance", False))
    min_dist = ff(summary.get("min_rel_dist"), 999.0)
    final_dist = ff(summary.get("final_rel_dist"), 999.0)
    dx = abs(ff(summary.get("odom_x_delta"), 0.0))
    yaw = abs(ff(summary.get("max_abs_mpc_yaw_rate"), 0.0))
    progress = ff(summary.get("progress_initial_minus_min"), 0.0)

    # Consistent with current Objective/RAM scorer.
    severe = final_dist > 0.75 or dx > 1.0
    approach = reached or min_dist <= 0.18
    yaw_sat = yaw >= 0.299

    # In Phase-B, yaw_rate may hit the command clamp during normal goal
    # steering. Treat yaw saturation as a penalty/risk feature, but do not
    # by itself prevent stable success if final distance and bounded drift are good.
    if stable or (reached and final_dist <= 0.30 and dx <= 0.80):
        return "stable_goal_reach_flat_locomotion"
    if severe:
        return "forward_walk_unreliable_on_soft_terrain"
    if approach and final_dist <= 0.60 and dx <= 1.0:
        return "approach_possible_but_post_reach_hold_needed"
    if progress > 0.08:
        return "cautious_probe_required"
    return "no_meaningful_progress"


def reward(summary):
    min_dist = ff(summary.get("min_rel_dist"), 999.0)
    final_dist = ff(summary.get("final_rel_dist"), 999.0)
    dx = abs(ff(summary.get("odom_x_delta"), 0.0))
    yaw = abs(ff(summary.get("max_abs_mpc_yaw_rate"), 0.0))
    progress = ff(summary.get("progress_initial_minus_min"), 0.0)
    reached = bb(summary.get("reached_stop_distance", False))

    sem = classify(summary)

    r = 0.0
    r += 2.0 * max(progress, 0.0)
    r -= 1.0 * min(final_dist, 5.0)
    r -= 0.5 * min(abs(dx), 5.0)
    r -= 0.5 if yaw >= 0.299 else 0.0

    if reached:
        r += 1.0
    if sem == "stable_goal_reach_flat_locomotion":
        r += 4.0
    elif sem == "approach_possible_but_post_reach_hold_needed":
        r += 0.5
    elif sem == "forward_walk_unreliable_on_soft_terrain":
        r -= 4.0
    elif sem == "no_meaningful_progress":
        r -= 1.0

    return r


def find_summary(run_dir):
    candidates = [
        Path(run_dir) / "episode" / "phase_b_summary.json",
        Path(run_dir) / "phase_b_summary.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    found = list(Path(run_dir).glob("**/phase_b_summary.json"))
    return found[0] if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    args = ap.parse_args()

    root = Path(args.sweep_root)
    manifest_path = root / "manifest_v0.json"
    manifest = json.load(open(manifest_path)) if manifest_path.is_file() else {"runs": []}

    rows = []
    for m in manifest.get("runs", []):
        if m.get("status") != "ok":
            continue

        run_dir = m["run_dir"]
        sp = find_summary(run_dir)
        if sp is None:
            continue

        s = json.load(open(sp))
        theta = m.get("theta", {})

        # Prefer actual summary values if present.
        action = {
            "vx_far": ff(s.get("vx_far"), ff(theta.get("vx_far"))),
            "vx_near": ff(s.get("vx_near"), ff(theta.get("vx_near"))),
            "goal_slow_distance": ff(s.get("goal_slow_distance"), ff(theta.get("goal_slow_distance"))),
            "goal_stop_distance": ff(s.get("goal_stop_distance"), ff(theta.get("goal_stop_distance"))),
            "body_height": ff(s.get("body_height"), ff(theta.get("body_height"))),
            "swing_clearance": ff(s.get("swing_clearance"), ff(theta.get("swing_clearance"))),
        }

        sem = classify(s)
        rew = reward(s)

        row = {
            "schema": "phase_b_theta_lite_dataset_row_v0",
            "source_run_dir": run_dir,
            "summary_json": str(sp),
            "world_name": m.get("world_name") or s.get("world_name"),
            "world_label": m.get("world_label"),
            "profile_name": m.get("profile_name"),
            "profile_family": m.get("profile_family"),
            "theta_action": action,
            "metrics": {
                "reached_stop_distance": bool(s.get("reached_stop_distance", False)),
                "min_rel_dist": ff(s.get("min_rel_dist"), 999.0),
                "final_rel_dist": ff(s.get("final_rel_dist"), 999.0),
                "progress_initial_minus_min": ff(s.get("progress_initial_minus_min"), 0.0),
                "odom_x_delta": ff(s.get("odom_x_delta"), 0.0),
                "max_abs_mpc_yaw_rate": ff(s.get("max_abs_mpc_yaw_rate"), 0.0),
            },
            "label": {
                "semantic": sem,
                "reward": rew,
                "stable_reached": sem == "stable_goal_reach_flat_locomotion",
                "future_invalid": sem == "forward_walk_unreliable_on_soft_terrain",
                "normal_walk_blocked": sem in (
                    "forward_walk_unreliable_on_soft_terrain",
                    "no_meaningful_progress",
                ),
            },
        }
        rows.append(row)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    sem_counts = Counter(r["label"]["semantic"] for r in rows)
    world_counts = Counter(r["world_name"] for r in rows)

    best_by_world = {}
    for r in rows:
        w = r["world_name"]
        if w not in best_by_world or r["label"]["reward"] > best_by_world[w]["label"]["reward"]:
            best_by_world[w] = r

    summary = {
        "schema": "phase_b_theta_lite_dataset_summary_v0",
        "sweep_root": str(root),
        "num_examples": len(rows),
        "semantic_counts": dict(sem_counts),
        "world_counts": dict(world_counts),
        "out_jsonl": args.out_jsonl,
        "best_by_world": {
            w: {
                "profile_name": r["profile_name"],
                "reward": r["label"]["reward"],
                "semantic": r["label"]["semantic"],
                "theta_action": r["theta_action"],
                "metrics": r["metrics"],
            }
            for w, r in best_by_world.items()
        },
    }

    Path(args.out_summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
