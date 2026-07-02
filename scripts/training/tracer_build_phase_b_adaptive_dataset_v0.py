#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


def fget(d, k, default=0.0):
    try:
        v = d.get(k, default)
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def bget(d, k, default=False):
    v = d.get(k, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compact_risk(risk):
    return {
        "ram_mismatch": fget(risk, "ram_mismatch"),
        "ram_uncertainty": fget(risk, "ram_uncertainty"),
        "recent_max_yaw_rate": fget(risk, "recent_max_yaw_rate"),
        "recent_slip_score": fget(risk, "recent_slip_score"),
        "body_stability_score": fget(risk, "body_stability_score"),
        "raw_max_abs_mpc_yaw_rate": fget(risk, "raw_max_abs_mpc_yaw_rate"),
        "yaw_p95_abs_rate": fget(risk, "yaw_p95_abs_rate"),
        "yaw_mean_abs_rate": fget(risk, "yaw_mean_abs_rate"),
        "yaw_saturation_fraction": fget(risk, "yaw_saturation_fraction"),
        "source": risk.get("source", ""),
        "reason": risk.get("reason", []),
    }


def compact_profile(selection, summary):
    selected = selection.get("selected", {})
    profile = selected.get("profile", {})

    # Fallback to summary because episode summaries include injected selected values.
    return {
        "selected_profile_name": summary.get("selected_profile_name", selected.get("name", "")),
        "selected_profile_score": fget(summary, "selected_profile_score", selected.get("score", 0.0)),
        "selected_profile_guard_reason": summary.get(
            "selected_profile_guard_reason",
            selected.get("guard_reason", ""),
        ),
        "vx_far": fget(profile, "vx_far", fget(summary, "vx_far")),
        "vx_near": fget(profile, "vx_near", fget(summary, "vx_near")),
        "goal_slow_distance": fget(profile, "goal_slow_distance", fget(summary, "goal_slow_distance")),
        "goal_stop_distance": fget(profile, "goal_stop_distance", fget(summary, "goal_stop_distance")),
        "goal_distance_ahead": fget(profile, "goal_distance_ahead", fget(summary, "goal_distance_ahead")),
        "risk_level": profile.get("risk_level", ""),
    }


def compact_outcome(summary):
    return {
        "reached_stop_distance": bget(summary, "reached_stop_distance"),
        "initial_rel_dist": fget(summary, "initial_rel_dist"),
        "final_rel_dist": fget(summary, "final_rel_dist"),
        "min_rel_dist": fget(summary, "min_rel_dist"),
        "progress_initial_minus_final": fget(summary, "progress_initial_minus_final"),
        "progress_initial_minus_min": fget(summary, "progress_initial_minus_min"),
        "odom_x_delta": fget(summary, "odom_x_delta"),
        "max_mpc_vx": fget(summary, "max_mpc_vx"),
        "max_abs_mpc_yaw_rate": fget(summary, "max_abs_mpc_yaw_rate"),
        "num_rows": fget(summary, "num_rows"),
        "num_valid_goal_rows": fget(summary, "num_valid_goal_rows"),
        "duration_sec": fget(summary, "duration_sec"),
    }


def outcome_score(summary):
    reached = bget(summary, "reached_stop_distance")
    progress = fget(summary, "progress_initial_minus_min")
    odom_dx = fget(summary, "odom_x_delta")
    min_dist = fget(summary, "min_rel_dist", 999.0)
    yaw = fget(summary, "max_abs_mpc_yaw_rate", 999.0)
    max_vx = fget(summary, "max_mpc_vx")

    score = 0.0
    score += 100.0 if reached else 0.0
    score += 40.0 * max(progress, 0.0)
    score += 8.0 * max(odom_dx, 0.0)
    score += 4.0 * max_vx
    score -= 20.0 * max(min_dist, 0.0)
    score -= 15.0 * max(yaw - 0.18, 0.0)
    score -= 30.0 * max(yaw - 0.28, 0.0)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive-root", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    args = ap.parse_args()

    root = Path(args.adaptive_root)
    adaptive_csv = root / "adaptive_summary.csv"

    if not adaptive_csv.is_file():
        raise FileNotFoundError(f"adaptive_summary.csv not found: {adaptive_csv}")

    with open(adaptive_csv, "r") as f:
        rows = list(csv.DictReader(f))

    examples = []

    for row in rows:
        ep = int(float(row["episode"]))
        risk_path = Path(row["risk_json"])
        summary_path = Path(row["summary_json"])

        ep_dir = summary_path.parent.parent
        selection_path = ep_dir / "selected_speed_profile.json"

        if not risk_path.is_file():
            print(f"[WARN] missing risk json: {risk_path}")
            continue
        if not summary_path.is_file():
            print(f"[WARN] missing summary json: {summary_path}")
            continue
        if not selection_path.is_file():
            print(f"[WARN] missing selection json: {selection_path}")
            selection = {"selected": {}}
        else:
            selection = load_json(selection_path)

        risk = load_json(risk_path)
        summary = load_json(summary_path)

        action = compact_profile(selection, summary)
        outcome = compact_outcome(summary)

        task = {
            "world_name": summary.get("world_name", "earth"),
            "model_name": summary.get("model_name", ""),
            "goal_distance_ahead": action["goal_distance_ahead"],
            "goal_stop_distance": action["goal_stop_distance"],
        }

        ex = {
            "episode_index": ep,
            "adaptive_root": str(root),
            "paths": {
                "risk_json": str(risk_path),
                "selection_json": str(selection_path),
                "summary_json": str(summary_path),
                "csv_path": summary.get("csv_path", ""),
            },
            "obs": {
                "risk_state": compact_risk(risk),
                "task": task,
            },
            "action": action,
            "outcome": outcome,
            "outcome_score": outcome_score(summary),
            "dataset_type": "phase_b_adaptive_pseudo_ram_v0",
        }

        examples.append(ex)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_jsonl, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, sort_keys=True) + "\n")

    summary_out = {
        "adaptive_root": str(root),
        "num_examples": len(examples),
        "out_jsonl": args.out_jsonl,
        "profiles": {},
    }

    for ex in examples:
        name = ex["action"]["selected_profile_name"]
        summary_out["profiles"].setdefault(name, 0)
        summary_out["profiles"][name] += 1

    with open(args.out_summary_json, "w") as f:
        json.dump(summary_out, f, indent=2, sort_keys=True)

    print(json.dumps(summary_out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
