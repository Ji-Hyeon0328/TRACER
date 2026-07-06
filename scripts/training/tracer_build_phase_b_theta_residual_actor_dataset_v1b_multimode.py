#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.runtime.tracer_run_phase_b_training_campaign_v1 import build_action_bank


THETA_KEYS = [
    "vx_far",
    "vx_near",
    "goal_slow_distance",
    "goal_stop_distance",
    "body_height",
    "swing_clearance",
]

TARGET_MODES = ["balanced", "reach", "stability"]

DELTA_SCALE = {
    "vx_far": 0.10,
    "vx_near": 0.05,
    "goal_slow_distance": 0.30,
    "goal_stop_distance": 0.15,
    "body_height": 0.05,
    "swing_clearance": 0.05,
}

WORLD_BETA_FALLBACK = {
    "earth": [0.600, 0.230, 0.170],
    "stairs_single": [0.461, 0.399, 0.141],
    "tracer_sponge_firm_flat": [0.140, 0.796, 0.063],
}

TARGET_ACTIONS = {
    "earth": {
        "balanced": "trot_mid",
        "reach": "trot_solid_fast",
        "stability": "trot_mid",
    },
    "stairs_single": {
        "balanced": "trot_solid_fast",
        "reach": "trot_solid_fast",
        "stability": "trot_mid",
    },
    "tracer_sponge_firm_flat": {
        "balanced": "sponge_v8b_reach_bias",
        "reach": "sponge_v8b_reach_stabilized",
        "stability": "sponge_slow_high_clear",
    },
}


def read_jsonl(p):
    rows = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)


def get_world_action(row):
    world = row.get("world") or row.get("world_name")
    action = row.get("action") or row.get("action_name") or row.get("selected_action_name")
    wa = row.get("world_action") or row.get("world::action")
    if (not world or not action) and isinstance(wa, str) and "::" in wa:
        world, action = wa.split("::", 1)
    return world, action


def extract_scores(task_beta_report):
    obj = load_json(task_beta_report)
    scores = {}

    for d in walk(obj):
        if not isinstance(d, dict):
            continue
        world, action = get_world_action(d)
        if not world or not action:
            continue
        score = d.get("score_task_beta")
        if score is None:
            score = d.get("target_task_beta")
        if score is None:
            continue
        try:
            scores[(world, action)] = float(score)
        except Exception:
            pass

    return scores


def extract_beta_by_world(beta_report):
    obj = load_json(beta_report)
    out = {}

    for d in walk(obj):
        if not isinstance(d, dict):
            continue
        world = d.get("world") or d.get("world_name")
        if not world:
            continue
        if all(k in d for k in ["beta_v", "beta_s", "beta_e"]):
            try:
                out[world] = [float(d["beta_v"]), float(d["beta_s"]), float(d["beta_e"])]
            except Exception:
                pass

    for k, v in WORLD_BETA_FALLBACK.items():
        out.setdefault(k, v)
    return out


def theta_vec(bank, action):
    if action not in bank:
        return None
    t = bank[action]
    if not all(k in t for k in THETA_KEYS):
        return None
    return {k: float(t[k]) for k in THETA_KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="data/phase_b_training_pipeline_v6_anchor_mixture_merged/phase_b_rollout_index_v1.jsonl")
    ap.add_argument("--task-beta-report", default="reports/phase_b_task_beta_reward_v1_anchor_mixture_merged.json")
    ap.add_argument("--beta-report", default="reports/phase_b_objective_selector_beta_v2_anchor_mixture_merged.json")
    ap.add_argument("--teacher-json", default="configs/phase_b_anchor_mixture_v8/ppo_warmstart_teacher_table_anchor_mixture_v8c_phase_scheduled.json")
    ap.add_argument("--out-dir", default="data/phase_b_theta_residual_actor_v1b_multimode")
    args = ap.parse_args()

    rollouts = read_jsonl(Path(args.rollouts))
    bank = build_action_bank(args.teacher_json)
    scores = extract_scores(args.task_beta_report)
    beta_by_world = extract_beta_by_world(args.beta_report)

    rows = []
    skipped = defaultdict(int)

    for r in rollouts:
        world, anchor_action = get_world_action(r)
        if not world or not anchor_action:
            skipped["missing_world_action"] += 1
            continue
        if world not in TARGET_ACTIONS:
            skipped["world_not_in_targets"] += 1
            continue

        anchor_theta = theta_vec(bank, anchor_action)
        if anchor_theta is None:
            skipped["missing_anchor_theta"] += 1
            continue

        beta = beta_by_world.get(world, WORLD_BETA_FALLBACK.get(world, [0.33, 0.33, 0.34]))
        anchor_score = scores.get((world, anchor_action), 0.0)

        for mode in TARGET_MODES:
            target_action = TARGET_ACTIONS[world][mode]
            target_theta = theta_vec(bank, target_action)
            if target_theta is None:
                skipped[f"missing_target_theta::{world}::{mode}::{target_action}"] += 1
                continue

            target_score = scores.get((world, target_action), 0.0)

            delta = {k: target_theta[k] - anchor_theta[k] for k in THETA_KEYS}
            delta_norm = {
                k: max(-1.0, min(1.0, delta[k] / DELTA_SCALE[k]))
                for k in THETA_KEYS
            }

            if mode == "balanced":
                mode_weight = 1.00
            elif mode == "reach":
                mode_weight = 0.75
            else:
                mode_weight = 0.75

            rows.append({
                "schema": "phase_b_theta_residual_actor_seed_v1b_multimode",
                "world": world,
                "target_mode": mode,
                "anchor_action": anchor_action,
                "target_action": target_action,
                "anchor_score_task_beta": anchor_score,
                "target_score_task_beta": target_score,
                "score_improvement": target_score - anchor_score,
                "train_weight": mode_weight,
                "beta_v": beta[0],
                "beta_s": beta[1],
                "beta_e": beta[2],
                **{f"anchor_{k}": anchor_theta[k] for k in THETA_KEYS},
                **{f"target_{k}": target_theta[k] for k in THETA_KEYS},
                **{f"delta_{k}": delta[k] for k in THETA_KEYS},
                **{f"delta_norm_{k}": delta_norm[k] for k in THETA_KEYS},
            })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "phase_b_theta_residual_actor_seed_v1b_multimode.jsonl", rows)

    by_target = defaultdict(int)
    by_anchor = defaultdict(int)
    for row in rows:
        by_target[f"{row['world']}::{row['target_mode']}::{row['target_action']}"] += 1
        by_anchor[f"{row['world']}::{row['target_mode']}::{row['anchor_action']}"] += 1

    report = {
        "schema": "phase_b_theta_residual_actor_dataset_report_v1b_multimode",
        "rows": len(rows),
        "skipped": dict(skipped),
        "target_actions": TARGET_ACTIONS,
        "by_target": dict(sorted(by_target.items())),
        "by_anchor": dict(sorted(by_anchor.items())),
    }

    with open(out_dir / "phase_b_theta_residual_actor_dataset_report_v1b_multimode.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    lines = []
    lines.append("# Phase-B θ Residual Actor Seed Dataset v1b Multimode")
    lines.append("")
    lines.append(f"- Rows: `{len(rows)}`")
    lines.append("")
    lines.append("## Target actions")
    lines.append("")
    lines.append("| world | mode | target_action | rows |")
    lines.append("|---|---|---|---:|")
    for k, n in report["by_target"].items():
        w, m, a = k.split("::", 2)
        lines.append(f"| {w} | {m} | {a} | {n} |")

    md = "\n".join(lines) + "\n"
    with open(out_dir / "phase_b_theta_residual_actor_dataset_report_v1b_multimode.md", "w") as f:
        f.write(md)

    print(md)
    print("[wrote]", out_dir / "phase_b_theta_residual_actor_seed_v1b_multimode.jsonl")


if __name__ == "__main__":
    main()
