#!/usr/bin/env python3

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


BAD_SEMANTICS = {
    "forward_walk_unreliable_on_soft_terrain",
    "no_meaningful_progress",
}

TRUSTED_SEMANTIC = "stable_goal_reach_flat_locomotion"


def mean(xs, default=0.0):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else default


def load_rows(paths):
    rows = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                r["_source_dataset"] = p
                rows.append(r)
    return rows


def theta_key(theta):
    """Group profiles by actual theta action, not by profile name."""
    keys = [
        "vx_far",
        "vx_near",
        "goal_slow_distance",
        "goal_stop_distance",
        "body_height",
        "swing_clearance",
    ]
    return tuple(round(float(theta.get(k, 0.0)), 4) for k in keys)


def theta_key_string(theta):
    k = theta_key(theta)
    names = ["vx", "near", "slow", "stop", "h", "clr"]
    return "::".join(f"{n}={v:.4f}" for n, v in zip(names, k))


def summarize_profile(rows):
    sems = [r["label"]["semantic"] for r in rows]
    rewards = [float(r["label"]["reward"]) for r in rows]
    metrics = [r["metrics"] for r in rows]

    n = len(rows)
    sem_counts = Counter(sems)
    severe_n = sum(1 for s in sems if s in BAD_SEMANTICS)
    stable_n = sum(1 for s in sems if s == TRUSTED_SEMANTIC)
    probe_n = sum(
        1
        for s in sems
        if s in {
            "cautious_probe_required",
            "approach_possible_but_post_reach_hold_needed",
        }
    )

    avg_reward = mean(rewards)
    avg_final = mean([m.get("final_rel_dist", 999.0) for m in metrics], 999.0)
    avg_min = mean([m.get("min_rel_dist", 999.0) for m in metrics], 999.0)
    avg_abs_dx = mean([abs(m.get("odom_x_delta", 0.0)) for m in metrics], 0.0)
    avg_yaw = mean([abs(m.get("max_abs_mpc_yaw_rate", 0.0)) for m in metrics], 0.0)

    severe_rate = severe_n / n if n else 1.0
    stable_rate = stable_n / n if n else 0.0
    probe_rate = probe_n / n if n else 0.0

    # Conservative trust rule.
    trusted = (
        n >= 1
        and stable_rate >= 0.60
        and severe_rate <= 0.20
        and avg_reward > 1.0
        and avg_final <= 0.30
        and avg_abs_dx <= 0.80
    )

    # Probe is useful for further local search, but not deployment.
    probe_candidate = (
        not trusted
        and severe_rate <= 0.33
        and probe_rate >= 0.50
        and avg_reward > -1.5
        and avg_final <= 0.60
        and avg_abs_dx <= 1.00
    )

    if trusted:
        status = "trusted_normal_walk"
    elif probe_candidate:
        status = "probe_candidate_requires_local_search"
    elif severe_rate >= 0.50:
        status = "blocked_requires_alternative_primitive"
    else:
        status = "untrusted_requires_more_search"

    row0 = rows[0]
    profile_counts = Counter(r.get("profile_name") for r in rows)
    family_counts = Counter(r.get("profile_family") for r in rows)
    primary_profile = profile_counts.most_common(1)[0][0]
    primary_family = family_counts.most_common(1)[0][0]

    return {
        "profile_name": primary_profile,
        "profile_names": dict(profile_counts),
        "profile_family": primary_family,
        "profile_families": dict(family_counts),
        "theta_key": theta_key_string(row0["theta_action"]),
        "theta_action": row0["theta_action"],
        "n": n,
        "semantic_counts": dict(sem_counts),
        "avg_reward": avg_reward,
        "avg_final_rel_dist": avg_final,
        "avg_min_rel_dist": avg_min,
        "avg_abs_odom_x_delta": avg_abs_dx,
        "avg_abs_yaw_rate": avg_yaw,
        "severe_rate": severe_rate,
        "stable_rate": stable_rate,
        "probe_rate": probe_rate,
        "selection_status": status,
        "trusted_normal_walk": trusted,
        "probe_candidate": probe_candidate,
        "normal_walk_blocked": status != "trusted_normal_walk",
        "requires_more_theta_search": status in {
            "probe_candidate_requires_local_search",
            "untrusted_requires_more_search",
        },
        "requires_alternative_primitive": status == "blocked_requires_alternative_primitive",
        "source_runs": [
            {
                "source_run_dir": r.get("source_run_dir"),
                "source_dataset": r.get("_source_dataset"),
                "semantic": r["label"]["semantic"],
                "reward": r["label"]["reward"],
                "metrics": r["metrics"],
            }
            for r in rows
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-jsonl", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    rows = load_rows(args.dataset_jsonl)

    grouped = defaultdict(list)
    for r in rows:
        key = (r["world_name"], theta_key(r["theta_action"]))
        grouped[key].append(r)

    by_world = defaultdict(list)
    for (world, _theta_key), rs in grouped.items():
        by_world[world].append(summarize_profile(rs))

    table = {}
    for world, profiles in by_world.items():
        trusted = [p for p in profiles if p["trusted_normal_walk"]]
        probes = [p for p in profiles if p["probe_candidate"]]

        if trusted:
            selected = sorted(trusted, key=lambda p: p["avg_reward"], reverse=True)[0]
            world_status = "trusted_normal_walk"
        elif probes:
            # A probe should only survive robust selection if it has repeated
            # non-severe evidence. A single lucky/probe run should not override
            # repeated severe local-search failures.
            robust_probes = [p for p in probes if p["n"] >= 2 and p["severe_rate"] <= 0.33]
            if robust_probes:
                selected = sorted(robust_probes, key=lambda p: p["avg_reward"], reverse=True)[0]
                world_status = "probe_candidate_requires_local_search"
            else:
                selected = sorted(profiles, key=lambda p: p["avg_reward"], reverse=True)[0]
                world_status = (
                    "blocked_requires_alternative_primitive"
                    if selected["severe_rate"] >= 0.50
                    else "untrusted_requires_more_search"
                )
        else:
            selected = sorted(profiles, key=lambda p: p["avg_reward"], reverse=True)[0]
            if selected["selection_status"] == "blocked_requires_alternative_primitive" or selected["severe_rate"] >= 0.50:
                world_status = "blocked_requires_alternative_primitive"
            else:
                world_status = "untrusted_requires_more_search"

        table[world] = {
            "world_name": world,
            "selection_status": world_status,
            "selected_profile_name": selected["profile_name"],
            "selected_profile_family": selected.get("profile_family"),
            "theta_action": selected["theta_action"],
            "trusted_normal_walk": world_status == "trusted_normal_walk",
            "probe_candidate": world_status == "probe_candidate_requires_local_search",
            "normal_walk_blocked": world_status != "trusted_normal_walk",
            "requires_more_theta_search": world_status in {
                "probe_candidate_requires_local_search",
                "untrusted_requires_more_search",
            },
            "requires_alternative_primitive": world_status == "blocked_requires_alternative_primitive",
            "selected_profile_summary": selected,
            "ranked_profiles": sorted(
                profiles,
                key=lambda p: p["avg_reward"],
                reverse=True,
            ),
        }

    out = {
        "schema": "phase_b_theta_lite_robust_teacher_table_v0",
        "source_dataset_jsonl": args.dataset_jsonl,
        "num_rows": len(rows),
        "num_worlds": len(table),
        "table": table,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
