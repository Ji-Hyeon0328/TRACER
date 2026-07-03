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


def classify_executed(summary):
    reached = bool(summary.get("reached_stop_distance", False))
    min_dist = ff(summary.get("min_rel_dist"), 999.0)
    final_dist = ff(summary.get("final_rel_dist"), 999.0)
    dx = abs(ff(summary.get("odom_x_delta"), 0.0))
    progress = ff(summary.get("progress_initial_minus_min"), 0.0)

    if reached and final_dist <= 0.30 and dx <= 0.80:
        return "stable_goal_reach_flat_locomotion"

    if final_dist > 0.75 or dx > 1.0:
        return "forward_walk_unreliable_on_soft_terrain"

    if reached or min_dist <= 0.18:
        return "approach_possible_but_post_reach_hold_needed"

    if progress > 0.08:
        return "cautious_probe_required"

    return "no_meaningful_progress"


def reward_executed(summary, semantic):
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
    elif semantic == "forward_walk_unreliable_on_soft_terrain":
        r -= 4.0
    elif semantic == "no_meaningful_progress":
        r -= 1.0

    return r


def load_json_or_none(path):
    if path and Path(path).is_file():
        return json.load(open(path))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard-root", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    args = ap.parse_args()

    root = Path(args.guard_root)
    manifest_path = root / "manifest_v0.json"

    if manifest_path.is_file():
        manifest = json.load(open(manifest_path))
        run_dirs = [Path(r["run_dir"]) for r in manifest.get("runs", [])]
    else:
        run_dirs = [Path(p) for p in sorted(glob.glob(str(root / "*")))]

    rows = []

    for run_dir in run_dirs:
        result = load_json_or_none(run_dir / "guarded_run_result_v0.json") or {}
        gate = load_json_or_none(run_dir / "runtime_gate_v0.json") or {}
        summary = load_json_or_none(run_dir / "episode" / "phase_b_summary.json") or {}

        world = (
            result.get("world_name")
            or gate.get("world_name")
            or summary.get("world_name")
            or run_dir.name
        )

        rollout_executed = bool(result.get("rollout_executed", summary.get("rollout_executed", False)))
        normal_walk_blocked = bool(result.get("normal_walk_blocked", summary.get("normal_walk_blocked", False)))

        theta = result.get("theta_action") or gate.get("theta_action") or summary.get("theta_action") or {}

        if not rollout_executed and normal_walk_blocked:
            semantic = "blocked_requires_alternative_primitive" if (
                result.get("requires_alternative_primitive")
                or summary.get("requires_alternative_primitive")
                or gate.get("requires_alternative_primitive")
            ) else "blocked_requires_more_theta_search"

            label = {
                "semantic": semantic,
                "reward": 0.0,
                "rollout_executed": False,
                "normal_walk_blocked": True,
                "allow_normal_walk": False,
                "prevented_bad_rollout": True,
                "future_invalid": True,
                "stable_reached": False,
                "requires_alternative_primitive": bool(
                    result.get("requires_alternative_primitive")
                    or summary.get("requires_alternative_primitive")
                    or gate.get("requires_alternative_primitive")
                ),
                "requires_more_theta_search": bool(
                    result.get("requires_more_theta_search")
                    or summary.get("requires_more_theta_search")
                    or gate.get("requires_more_theta_search")
                ),
            }

            metrics = {
                "reached_stop_distance": None,
                "min_rel_dist": None,
                "final_rel_dist": None,
                "progress_initial_minus_min": None,
                "odom_x_delta": None,
                "max_abs_mpc_yaw_rate": None,
            }

        else:
            semantic = classify_executed(summary)
            rew = reward_executed(summary, semantic)

            label = {
                "semantic": semantic,
                "reward": rew,
                "rollout_executed": True,
                "normal_walk_blocked": False,
                "allow_normal_walk": True,
                "prevented_bad_rollout": False,
                "future_invalid": semantic in {
                    "forward_walk_unreliable_on_soft_terrain",
                    "no_meaningful_progress",
                },
                "stable_reached": semantic == "stable_goal_reach_flat_locomotion",
                "requires_alternative_primitive": False,
                "requires_more_theta_search": False,
            }

            metrics = {
                "reached_stop_distance": bool(summary.get("reached_stop_distance", False)),
                "min_rel_dist": ff(summary.get("min_rel_dist"), 999.0),
                "final_rel_dist": ff(summary.get("final_rel_dist"), 999.0),
                "progress_initial_minus_min": ff(summary.get("progress_initial_minus_min"), 0.0),
                "odom_x_delta": ff(summary.get("odom_x_delta"), 0.0),
                "max_abs_mpc_yaw_rate": ff(summary.get("max_abs_mpc_yaw_rate"), 0.0),
            }

        row = {
            "schema": "phase_b_guarded_outcome_dataset_row_v0",
            "source_run_dir": str(run_dir),
            "world_name": world,
            "selected_profile_name": (
                result.get("selected_profile_name")
                or gate.get("selected_profile_name")
                or summary.get("selected_profile_name")
            ),
            "selection_status": (
                result.get("selection_status")
                or gate.get("selection_status")
                or summary.get("selection_status")
            ),
            "theta_action": theta,
            "runtime_gate_json": str(run_dir / "runtime_gate_v0.json"),
            "guarded_result_json": str(run_dir / "guarded_run_result_v0.json"),
            "summary_json": str(run_dir / "episode" / "phase_b_summary.json"),
            "metrics": metrics,
            "label": label,
        }

        rows.append(row)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    summary = {
        "schema": "phase_b_guarded_outcome_dataset_summary_v0",
        "guard_root": str(root),
        "num_examples": len(rows),
        "world_counts": dict(Counter(r["world_name"] for r in rows)),
        "semantic_counts": dict(Counter(r["label"]["semantic"] for r in rows)),
        "num_rollout_executed": sum(1 for r in rows if r["label"]["rollout_executed"]),
        "num_normal_walk_blocked": sum(1 for r in rows if r["label"]["normal_walk_blocked"]),
        "num_prevented_bad_rollout": sum(1 for r in rows if r["label"]["prevented_bad_rollout"]),
        "out_jsonl": args.out_jsonl,
    }

    Path(args.out_summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
