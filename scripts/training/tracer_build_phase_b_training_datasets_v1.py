#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


WORLD_VOCAB = [
    "earth",
    "stairs_single",
    "tracer_sponge_firm_flat",
]


ACTION_FIELDS = [
    "vx_far",
    "vx_near",
    "goal_slow_distance",
    "goal_stop_distance",
    "body_height",
    "swing_clearance",
]


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


def bb(x):
    return bool(x)


def world_features(world):
    return {
        "world_onehot:earth": 1.0 if world == "earth" else 0.0,
        "world_onehot:stairs_single": 1.0 if world == "stairs_single" else 0.0,
        "world_onehot:tracer_sponge_firm_flat": 1.0 if world == "tracer_sponge_firm_flat" else 0.0,
        "terrain_is_flat": 1.0 if world in ["earth", "tracer_sponge_firm_flat"] else 0.0,
        "terrain_is_stairs": 1.0 if world == "stairs_single" else 0.0,
        "terrain_is_soft_contact": 1.0 if world == "tracer_sponge_firm_flat" else 0.0,
        "terrain_roughness_proxy": 0.45 if world == "stairs_single" else 0.05 if world == "tracer_sponge_firm_flat" else 0.0,
        "terrain_slip_proxy": 0.03 if world == "tracer_sponge_firm_flat" else 0.02 if world == "stairs_single" else 0.0,
        "terrain_stiffness_log_proxy": 8.0 if world == "tracer_sponge_firm_flat" else 12.0,
    }


def row_features(row):
    world = row["world_name"]
    theta = row.get("theta_action", {}) or {}

    feat = {}
    feat.update(world_features(world))

    for k in ACTION_FIELDS:
        feat[f"theta:{k}"] = ff(theta.get(k))

    m = row.get("metrics", {})
    c = row.get("reward_components", {})

    feat.update({
        "metric:reached": 1.0 if c.get("reached_stop_distance") else 0.0,
        "metric:min_rel_dist": ff(c.get("min_rel_dist")),
        "metric:final_rel_dist": ff(c.get("final_rel_dist")),
        "metric:odom_x_delta_abs": ff(c.get("odom_x_delta_abs")),
        "metric:max_abs_mpc_yaw_rate": ff(c.get("max_abs_mpc_yaw_rate")),
        "reward:reach_reward": ff(c.get("reach_reward")),
        "reward:hold_reward": ff(c.get("hold_reward")),
    })

    return feat


def score_for_preference(row):
    c = row.get("reward_components", {})
    reach = ff(c.get("reach_reward"))
    hold = ff(c.get("hold_reward"))
    reached = 1.0 if c.get("reached_stop_distance") else 0.0
    final_dist = ff(c.get("final_rel_dist"))
    min_dist = ff(c.get("min_rel_dist"))

    # Seed preference score. This is not the final reward model;
    # it bootstraps preference pairs for Objective Selector training.
    return reach + 0.30 * hold + 2.0 * reached - 0.25 * final_dist - 0.25 * min_dist


def ram_labels(row):
    c = row.get("reward_components", {})
    reached = bb(c.get("reached_stop_distance"))
    min_dist = ff(c.get("min_rel_dist"), 999.0)
    final_dist = ff(c.get("final_rel_dist"), 999.0)
    dx = ff(c.get("odom_x_delta_abs"), 0.0)
    reach_reward = ff(c.get("reach_reward"), 0.0)
    hold_reward = ff(c.get("hold_reward"), 0.0)

    high_drift = final_dist > 1.0 or dx > 1.5
    low_progress = (not reached) and min_dist > 0.20
    bad = (not reached) or high_drift or hold_reward < -1.0
    good = reached and final_dist < 0.5 and hold_reward > 0.0

    risk = 0.0
    risk += 0.40 if not reached else 0.0
    risk += 0.35 if high_drift else 0.0
    risk += 0.15 if low_progress else 0.0
    risk += 0.10 if hold_reward < 0.0 else 0.0
    risk = max(0.0, min(1.0, risk))

    return {
        "approach_success": reached,
        "future_low_progress": bool(low_progress),
        "future_high_drift": bool(high_drift),
        "future_bad_locomotion": bool(bad),
        "future_good_locomotion": bool(good),
        "future_risk_proxy": risk,
        "reach_reward": reach_reward,
        "hold_reward": hold_reward,
        "min_rel_dist": min_dist,
        "final_rel_dist": final_dist,
        "odom_x_delta_abs": dx,
    }


def read_rollout_rows(root):
    rows = []

    for p in sorted(Path(root).rglob("ppo_policy_episode_result_v0.json")):
        try:
            r = load_json(p)
        except Exception:
            continue

        if r.get("status") != "ok":
            continue

        world = r.get("world_name")
        action = r.get("selected_action_name")

        if not world or not action:
            continue

        row = {
            "schema": "phase_b_training_rollout_row_v1",
            "source_path": str(p),
            "run_dir": str(p.parent),
            "world_name": world,
            "action_name": action,
            "theta_action": r.get("theta_action", {}),
            "metrics": r.get("metrics", {}),
            "reward_components": r.get("reward_components", {}),
        }
        row["preference_score_seed"] = score_for_preference(row)
        row["features"] = row_features(row)
        row["ram_labels"] = ram_labels(row)
        rows.append(row)

    return rows


def make_preference_pairs(rows, margin):
    pairs = []

    by_world = {}
    for r in rows:
        by_world.setdefault(r["world_name"], []).append(r)

    for world, rs in by_world.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                sa = ff(a["preference_score_seed"])
                sb = ff(b["preference_score_seed"])

                if abs(sa - sb) < margin:
                    continue

                winner, loser = (a, b) if sa > sb else (b, a)

                pairs.append({
                    "schema": "phase_b_objective_preference_pair_v1",
                    "world_name": world,
                    "winner_source": winner["source_path"],
                    "loser_source": loser["source_path"],
                    "winner_action": winner["action_name"],
                    "loser_action": loser["action_name"],
                    "winner_score_seed": winner["preference_score_seed"],
                    "loser_score_seed": loser["preference_score_seed"],
                    "score_gap": abs(sa - sb),
                    "winner_features": winner["features"],
                    "loser_features": loser["features"],
                    "note": "Seed pair for bootstrap preference/self-IRL Objective Selector training."
                })

    return pairs


def make_ppo_rows(rows):
    out = []
    action_vocab = sorted(set(r["action_name"] for r in rows))
    action_to_i = {a: i for i, a in enumerate(action_vocab)}

    for r in rows:
        out.append({
            "schema": "phase_b_discrete_theta_lite_ppo_episode_row_v1",
            "source_path": r["source_path"],
            "world_name": r["world_name"],
            "action_name": r["action_name"],
            "action_index": action_to_i[r["action_name"]],
            "action_vocab": action_vocab,
            "observation_features": {
                k: v for k, v in r["features"].items()
                if k.startswith("world_onehot") or k.startswith("terrain_")
            },
            "theta_action": r["theta_action"],
            "reward_seed": r["preference_score_seed"],
            "reward_components": r["reward_components"],
            "ram_labels": r["ram_labels"],
        })

    return out


def write_csv_summary(p, rows):
    fields = [
        "source_path",
        "world_name",
        "action_name",
        "preference_score_seed",
        "reached",
        "min_rel_dist",
        "final_rel_dist",
        "odom_x_delta_abs",
        "reach_reward",
        "hold_reward",
        "future_risk_proxy",
        "future_bad_locomotion",
        "future_good_locomotion",
    ]

    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            c = r["reward_components"]
            lab = r["ram_labels"]
            w.writerow({
                "source_path": r["source_path"],
                "world_name": r["world_name"],
                "action_name": r["action_name"],
                "preference_score_seed": r["preference_score_seed"],
                "reached": c.get("reached_stop_distance"),
                "min_rel_dist": c.get("min_rel_dist"),
                "final_rel_dist": c.get("final_rel_dist"),
                "odom_x_delta_abs": c.get("odom_x_delta_abs"),
                "reach_reward": c.get("reach_reward"),
                "hold_reward": c.get("hold_reward"),
                "future_risk_proxy": lab.get("future_risk_proxy"),
                "future_bad_locomotion": lab.get("future_bad_locomotion"),
                "future_good_locomotion": lab.get("future_good_locomotion"),
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pair-margin", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rollout_rows(args.result_root)
    pairs = make_preference_pairs(rows, margin=args.pair_margin)
    ppo_rows = make_ppo_rows(rows)

    ram_rows = []
    for r in rows:
        ram_rows.append({
            "schema": "phase_b_ram_teacher_student_episode_label_row_v1",
            "source_path": r["source_path"],
            "world_name": r["world_name"],
            "action_name": r["action_name"],
            "theta_action": r["theta_action"],
            "input_features": r["features"],
            "labels": r["ram_labels"],
        })

    write_jsonl(out_dir / "phase_b_rollout_index_v1.jsonl", rows)
    write_csv_summary(out_dir / "phase_b_rollout_index_v1.csv", rows)
    write_jsonl(out_dir / "phase_b_objective_preference_pairs_v1.jsonl", pairs)
    write_jsonl(out_dir / "phase_b_ram_teacher_student_episode_labels_v1.jsonl", ram_rows)
    write_jsonl(out_dir / "phase_b_discrete_theta_lite_ppo_rows_v1.jsonl", ppo_rows)

    by_world_action = {}
    for r in rows:
        key = f"{r['world_name']}::{r['action_name']}"
        by_world_action.setdefault(key, {
            "n": 0,
            "mean_score": 0.0,
            "reach_count": 0,
            "mean_risk": 0.0,
        })
        g = by_world_action[key]
        g["n"] += 1
        g["mean_score"] += r["preference_score_seed"]
        g["reach_count"] += 1 if r["reward_components"].get("reached_stop_distance") else 0
        g["mean_risk"] += r["ram_labels"]["future_risk_proxy"]

    for g in by_world_action.values():
        n = max(g["n"], 1)
        g["mean_score"] /= n
        g["reach_rate"] = g["reach_count"] / n
        g["mean_risk"] /= n

    report = {
        "schema": "phase_b_training_dataset_build_report_v1",
        "result_root": args.result_root,
        "out_dir": str(out_dir),
        "num_rollouts": len(rows),
        "num_preference_pairs": len(pairs),
        "num_ram_rows": len(ram_rows),
        "num_ppo_rows": len(ppo_rows),
        "by_world_action": by_world_action,
        "outputs": {
            "rollout_index_jsonl": str(out_dir / "phase_b_rollout_index_v1.jsonl"),
            "rollout_index_csv": str(out_dir / "phase_b_rollout_index_v1.csv"),
            "objective_preference_pairs_jsonl": str(out_dir / "phase_b_objective_preference_pairs_v1.jsonl"),
            "ram_labels_jsonl": str(out_dir / "phase_b_ram_teacher_student_episode_labels_v1.jsonl"),
            "ppo_rows_jsonl": str(out_dir / "phase_b_discrete_theta_lite_ppo_rows_v1.jsonl"),
        }
    }

    save_json(out_dir / "phase_b_training_dataset_build_report_v1.json", report)

    lines = []
    lines.append("# Phase-B Training Dataset Build Report")
    lines.append("")
    lines.append(f"- Result root: `{args.result_root}`")
    lines.append(f"- Output dir: `{out_dir}`")
    lines.append(f"- Rollouts: `{len(rows)}`")
    lines.append(f"- Objective preference pairs: `{len(pairs)}`")
    lines.append(f"- RAM label rows: `{len(ram_rows)}`")
    lines.append(f"- PPO rows: `{len(ppo_rows)}`")
    lines.append("")
    lines.append("| world::action | n | reach_rate | mean_score | mean_risk |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, g in sorted(by_world_action.items()):
        lines.append(
            f"| {key} | {g['n']} | {g['reach_rate']:.3f} | {g['mean_score']:.3f} | {g['mean_risk']:.3f} |"
        )
    lines.append("")
    lines.append("## Dataset roles")
    lines.append("")
    lines.append("- `phase_b_objective_preference_pairs_v1.jsonl`: bootstrap preference/self-IRL Objective Selector seed pairs.")
    lines.append("- `phase_b_ram_teacher_student_episode_labels_v1.jsonl`: RAM teacher-student seed labels from rollout outcomes.")
    lines.append("- `phase_b_discrete_theta_lite_ppo_rows_v1.jsonl`: discrete theta-lite PPO episode-level seed rows.")
    lines.append("")
    lines.append("This is a dataset seed pipeline. It does not replace later window-level RAM training or β/RAM-conditioned PPO, but it closes the collection-to-training-data loop.")

    (out_dir / "phase_b_training_dataset_build_report_v1.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print("[wrote]", out_dir / "phase_b_training_dataset_build_report_v1.json")
    print("[wrote]", out_dir / "phase_b_training_dataset_build_report_v1.md")


if __name__ == "__main__":
    main()
