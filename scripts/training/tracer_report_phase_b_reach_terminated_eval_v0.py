#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.training.tracer_compute_phase_b_tracer_reward_v0 import (
    compute_tracer_proxy_reward,
    load_json,
)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def maybe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def compute_time_to_reach(summary):
    csv_path = summary.get("csv_path")
    stop = maybe_float(summary.get("stop_distance", summary.get("goal_stop_distance", 0.15)), 0.15)
    if not csv_path:
        return None

    p = Path(csv_path)
    if not p.exists():
        return None

    try:
        with open(p, "r") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None

    if not rows:
        return None

    time_keys = ["t", "time", "time_sec", "elapsed_sec", "stamp_sec"]
    dist_keys = ["rel_dist", "relative_goal_dist", "goal_dist", "rel_goal_dist"]

    time_key = next((k for k in time_keys if k in rows[0]), None)
    dist_key = next((k for k in dist_keys if k in rows[0]), None)
    has_xy = "rel_x" in rows[0] and "rel_y" in rows[0]

    if not dist_key and not has_xy:
        return None

    t0 = maybe_float(rows[0].get(time_key, 0.0), 0.0) if time_key else 0.0

    for idx, r in enumerate(rows):
        if dist_key:
            d = maybe_float(r.get(dist_key), 999.0)
        else:
            rx = maybe_float(r.get("rel_x"), 999.0)
            ry = maybe_float(r.get("rel_y"), 0.0)
            d = math.sqrt(rx * rx + ry * ry)

        if d <= stop:
            if time_key:
                return max(0.0, maybe_float(r.get(time_key), 0.0) - t0)
            dur = maybe_float(summary.get("duration_sec"), 0.0)
            return dur * idx / max(1, len(rows) - 1)

    return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize_group(rows):
    n = len(rows)
    reached = [bool(r["reached_stop_distance"]) for r in rows]

    out = {
        "n": n,
        "reach_rate": sum(reached) / max(1, n),
        "mean_min_rel_dist": mean([r["min_rel_dist"] for r in rows]),
        "mean_final_rel_dist": mean([r["final_rel_dist"] for r in rows]),
        "mean_post_reach_drift": mean([r["post_reach_drift"] for r in rows]),
        "mean_reach_reward": mean([r["reach_reward"] for r in rows]),
        "mean_hold_reward": mean([r["hold_reward"] for r in rows]),
        "mean_tracer_proxy_reward": mean([r["tracer_proxy_reward"] for r in rows]),
        "mean_R_v": mean([r["tracer_terms"].get("R_v") for r in rows]),
        "mean_R_s": mean([r["tracer_terms"].get("R_s") for r in rows]),
        "mean_R_e": mean([r["tracer_terms"].get("R_e") for r in rows]),
        "mean_time_to_reach": mean([r["time_to_reach"] for r in rows if r["reached_stop_distance"]]),
        "best_min_rel_dist": min([r["min_rel_dist"] for r in rows]) if rows else None,
        "best_final_rel_dist": min([r["final_rel_dist"] for r in rows]) if rows else None,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--reward-config", default="configs/phase_b_reward_v0/tracer_slide_reward_proxy_v0.json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--policy-name", default="tracer_proxy_ppo")
    args = ap.parse_args()

    result_root = Path(args.result_root)
    reward_cfg = load_json(args.reward_config)

    rows = []
    skipped = {}

    def skip(k):
        skipped[k] = skipped.get(k, 0) + 1

    for rp in sorted(result_root.glob("**/ppo_policy_episode_result_v0.json")):
        try:
            r = load_json(rp)
        except Exception:
            skip("bad_result_json")
            continue

        sp = rp.parent / "phase_b_summary.json"
        if not sp.exists():
            skip("missing_summary")
            continue

        try:
            s = load_json(sp)
        except Exception:
            skip("bad_summary_json")
            continue

        c = r.get("reward_components", {})
        tr = compute_tracer_proxy_reward(s, reward_cfg)

        min_d = maybe_float(c.get("min_rel_dist", s.get("min_rel_dist")), 999.0)
        final_d = maybe_float(c.get("final_rel_dist", s.get("final_rel_dist")), 999.0)

        row = {
            "source": str(rp),
            "policy_name": args.policy_name,
            "world_name": r.get("world_name", s.get("world_name")),
            "selected_action_name": r.get("selected_action_name"),
            "status": r.get("status"),
            "reached_stop_distance": bool(c.get("reached_stop_distance", s.get("reached_stop_distance", False))),
            "min_rel_dist": min_d,
            "final_rel_dist": final_d,
            "post_reach_drift": max(0.0, final_d - min_d),
            "reach_reward": maybe_float(c.get("reach_reward"), 0.0),
            "hold_reward": maybe_float(c.get("hold_reward"), 0.0),
            "time_to_reach": compute_time_to_reach(s),
            "tracer_proxy_reward": tr["reward"],
            "tracer_terms": tr["terms"],
            "tracer_beta": tr["beta"],
        }
        rows.append(row)

    by_world = {}
    by_action = {}
    by_world_action = {}

    for row in rows:
        by_world.setdefault(row["world_name"], []).append(row)
        by_action.setdefault(row["selected_action_name"], []).append(row)
        key = f'{row["world_name"]}::{row["selected_action_name"]}'
        by_world_action.setdefault(key, []).append(row)

    report = {
        "schema": "phase_b_reach_terminated_eval_report_v0",
        "result_root": str(result_root),
        "policy_name": args.policy_name,
        "reward_config": args.reward_config,
        "interpretation": {
            "main_claim": "Reach-terminated evaluation treats goal-neighborhood arrival as the high-level planner success condition.",
            "main_metrics": [
                "reached_stop_distance",
                "min_rel_dist",
                "time_to_reach",
                "R_v"
            ],
            "auxiliary_limitation_metrics": [
                "final_rel_dist",
                "post_reach_drift",
                "hold_reward",
                "R_s"
            ],
            "controller_scope": "Low-level controller is frozen. Post-reach hold failures are reported as limitations rather than high-level traversal failures."
        },
        "num_rows": len(rows),
        "skipped": skipped,
        "overall": summarize_group(rows),
        "by_world": {k: summarize_group(v) for k, v in sorted(by_world.items())},
        "by_action": {k: summarize_group(v) for k, v in sorted(by_action.items())},
        "by_world_action": {k: summarize_group(v) for k, v in sorted(by_world_action.items())},
        "rows": rows,
    }

    save_json(Path(args.out_json), report)

    lines = []
    lines.append("# Phase-B Reach-Terminated Evaluation")
    lines.append("")
    lines.append(f"- Policy: `{args.policy_name}`")
    lines.append(f"- Result root: `{result_root}`")
    lines.append("- Controller: frozen low-level controller")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Main success means the robot reached the goal neighborhood at any point during the episode.")
    lines.append("- Final distance and hold reward are reported as auxiliary post-reach stability limitations.")
    lines.append("- This separates high-level traversal from low-level hold/stabilization.")
    lines.append("")
    lines.append("## By World")
    lines.append("")
    lines.append("| world | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s | mean_reward |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for world, g in report["by_world"].items():
        lines.append(
            f"| {world} | {g['n']} | {g['reach_rate']:.3f} | "
            f"{g['mean_min_rel_dist']:.3f} | {g['mean_final_rel_dist']:.3f} | "
            f"{g['mean_post_reach_drift']:.3f} | {g['mean_R_v']:.3f} | "
            f"{g['mean_R_s']:.3f} | {g['mean_tracer_proxy_reward']:.3f} |"
        )

    lines.append("")
    lines.append("## By World and Action")
    lines.append("")
    lines.append("| world::action | n | reach_rate | mean_min_dist | mean_final_dist | mean_post_reach_drift | mean_R_v | mean_R_s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for key, g in report["by_world_action"].items():
        lines.append(
            f"| {key} | {g['n']} | {g['reach_rate']:.3f} | "
            f"{g['mean_min_rel_dist']:.3f} | {g['mean_final_rel_dist']:.3f} | "
            f"{g['mean_post_reach_drift']:.3f} | {g['mean_R_v']:.3f} | "
            f"{g['mean_R_s']:.3f} |"
        )

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print(json.dumps(report["by_world"], indent=2, sort_keys=True))
    print()
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
