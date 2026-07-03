#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA_DATASET = "phase_b_ppo_warmstart_dataset_v0"
SCHEMA_SUMMARY = "phase_b_ppo_warmstart_summary_v0"
SCHEMA_TEACHER = "phase_b_ppo_warmstart_teacher_table_v0"


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def ff(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def softmax(xs, temperature=1.0):
    if not xs:
        return []
    t = max(float(temperature), 1e-6)
    m = max(xs)
    exps = [math.exp((x - m) / t) for x in xs]
    z = sum(exps)
    if z <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [e / z for e in exps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", default="artifacts")
    ap.add_argument("--out-dir", default="configs/phase_b_ppo_warmstart_v0")
    ap.add_argument(
        "--worlds",
        nargs="+",
        default=["earth", "stairs_single", "tracer_sponge_firm_flat"],
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-ok-only", action="store_true", default=True)
    ap.add_argument("--policy-json", default="configs/phase_b_theta_lite_rl_v0/current_bandit_policy.json")
    args = ap.parse_args()

    artifact_root = Path(args.artifact_root)
    out_dir = Path(args.out_dir)
    worlds = set(args.worlds)

    result_paths = sorted(
        artifact_root.glob("phase_b_theta_lite_bandit_worldset_v0_*/*/bandit_episode_result_v0.json")
    )

    rows = []
    skipped = Counter()

    for p in result_paths:
        try:
            r = load_json(p)
        except Exception:
            skipped["json_load_failed"] += 1
            continue

        world = r.get("world_name")
        if world not in worlds:
            skipped["world_not_selected"] += 1
            continue

        status = r.get("status")
        if args.min_ok_only and status != "ok":
            skipped["not_ok"] += 1
            continue

        action_name = r.get("selected_action_name")
        theta_action = r.get("theta_action") or {}

        if not action_name:
            skipped["missing_action"] += 1
            continue

        c = r.get("reward_components") or {}
        scalar_reward = ff(r.get("reward"), 0.0)
        reach_reward = ff(c.get("reach_reward"), ff(r.get("bandit_update_reward"), scalar_reward))
        hold_reward = ff(c.get("hold_reward"), 0.0)

        reached = bool(c.get("reached_stop_distance", r.get("metrics", {}).get("reached_stop_distance", False)))
        min_rel_dist = ff(c.get("min_rel_dist"), ff(r.get("metrics", {}).get("min_rel_dist"), 999.0))
        final_rel_dist = ff(c.get("final_rel_dist"), ff(r.get("metrics", {}).get("final_rel_dist"), 999.0))
        odom_x_delta_abs = ff(c.get("odom_x_delta_abs"), abs(ff(r.get("metrics", {}).get("odom_x_delta"), 0.0)))
        max_abs_yaw = ff(c.get("max_abs_mpc_yaw_rate"), ff(r.get("metrics", {}).get("max_abs_mpc_yaw_rate"), 0.0))

        semantic = r.get("semantic", "unknown")

        # Dataset labels for downstream supervised warm-start / PPO auxiliary losses.
        reach_success = bool(reached or reach_reward >= 5.0 or min_rel_dist <= 0.16)
        hold_success = bool(reached and final_rel_dist <= 0.30 and odom_x_delta_abs <= 0.80)

        # Use non-negative reach reward as BC/sample weight.
        bc_weight = max(0.05, min(1.0, reach_reward / 8.0))

        row = {
            "schema": SCHEMA_DATASET,
            "source_result_json": str(p),
            "world_name": world,
            "context_id": world,
            "action_name": action_name,
            "theta_action": theta_action,
            "status": status,
            "semantic": semantic,
            "scalar_reward": scalar_reward,
            "bandit_update_reward": ff(r.get("bandit_update_reward"), reach_reward),
            "reach_reward": reach_reward,
            "hold_reward": hold_reward,
            "reached_stop_distance": reached,
            "reach_success": reach_success,
            "hold_success": hold_success,
            "min_rel_dist": min_rel_dist,
            "final_rel_dist": final_rel_dist,
            "odom_x_delta_abs": odom_x_delta_abs,
            "max_abs_mpc_yaw_rate": max_abs_yaw,
            "bc_weight": bc_weight,
        }
        rows.append(row)

    # Aggregate world-action statistics.
    groups = defaultdict(list)
    for row in rows:
        groups[(row["world_name"], row["action_name"])].append(row)

    world_action_stats = defaultdict(list)
    for (world, action), rs in groups.items():
        reach_vals = [x["reach_reward"] for x in rs]
        hold_vals = [x["hold_reward"] for x in rs]
        sem_counts = Counter(x["semantic"] for x in rs)
        theta_action = rs[-1].get("theta_action", {})

        stat = {
            "world_name": world,
            "action_name": action,
            "theta_action": theta_action,
            "n": len(rs),
            "mean_reach_reward": statistics.mean(reach_vals),
            "best_reach_reward": max(reach_vals),
            "last_reach_reward": reach_vals[-1],
            "std_reach_reward": statistics.pstdev(reach_vals) if len(reach_vals) > 1 else 0.0,
            "mean_hold_reward": statistics.mean(hold_vals),
            "best_hold_reward": max(hold_vals),
            "reach_success_rate": sum(1 for x in rs if x["reach_success"]) / len(rs),
            "hold_success_rate": sum(1 for x in rs if x["hold_success"]) / len(rs),
            "semantic_counts": dict(sem_counts),
        }
        world_action_stats[world].append(stat)

    teacher = {
        "schema": SCHEMA_TEACHER,
        "source_policy_json": args.policy_json,
        "worlds": {},
    }

    for world, stats in world_action_stats.items():
        ranked = sorted(
            stats,
            key=lambda x: (
                x["mean_reach_reward"],
                x["reach_success_rate"],
                x["best_reach_reward"],
                x["n"],
            ),
            reverse=True,
        )

        top = ranked[: args.top_k]
        probs = softmax([x["mean_reach_reward"] for x in top], temperature=1.0)

        teacher["worlds"][world] = {
            "top_k": args.top_k,
            "num_actions_seen": len(ranked),
            "ranked_actions": ranked,
            "ppo_action_prior": [
                {
                    "action_name": x["action_name"],
                    "prob": probs[i],
                    "mean_reach_reward": x["mean_reach_reward"],
                    "mean_hold_reward": x["mean_hold_reward"],
                    "reach_success_rate": x["reach_success_rate"],
                    "theta_action": x["theta_action"],
                }
                for i, x in enumerate(top)
            ],
        }

    summary = {
        "schema": SCHEMA_SUMMARY,
        "num_rows": len(rows),
        "num_result_json_files_scanned": len(result_paths),
        "selected_worlds": sorted(worlds),
        "skipped": dict(skipped),
        "rows_by_world": dict(Counter(x["world_name"] for x in rows)),
        "rows_by_action": dict(Counter(x["action_name"] for x in rows)),
        "top_action_by_world": {
            w: teacher["worlds"][w]["ppo_action_prior"][0] if teacher["worlds"][w]["ppo_action_prior"] else None
            for w in teacher["worlds"]
        },
    }

    write_jsonl(out_dir / "ppo_warmstart_dataset.jsonl", rows)
    save_json(out_dir / "ppo_warmstart_summary.json", summary)
    save_json(out_dir / "ppo_warmstart_teacher_table.json", teacher)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
