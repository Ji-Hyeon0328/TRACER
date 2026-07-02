#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path


def fget(d, k, default=0.0):
    try:
        v = d.get(k, default)
        if v is None or v == "":
            return default
        x = float(v)
        if not math.isfinite(x):
            return default
        return x
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


def outcome_score(summary):
    reached = bget(summary, "reached_stop_distance")
    progress = fget(summary, "progress_initial_minus_min")
    odom_dx = fget(summary, "odom_x_delta")
    min_dist = fget(summary, "min_rel_dist", 999.0)
    yaw = fget(summary, "max_abs_mpc_yaw_rate", 999.0)
    vx = fget(summary, "max_mpc_vx")

    score = 0.0
    score += 100.0 if reached else 0.0
    score += 40.0 * max(progress, 0.0)
    score += 8.0 * max(odom_dx, 0.0)
    score += 3.0 * max(vx, 0.0)
    score -= 20.0 * max(min_dist, 0.0)
    score -= 15.0 * max(yaw - 0.18, 0.0)
    score -= 30.0 * max(yaw - 0.28, 0.0)
    return score


def risk_template(risk_name):
    if risk_name == "low":
        return {
            "ram_mismatch": 0.04,
            "ram_uncertainty": 0.04,
            "recent_max_yaw_rate": 0.08,
            "recent_slip_score": 0.02,
            "body_stability_score": 0.96,
            "yaw_p95_abs_rate": 0.06,
            "yaw_mean_abs_rate": 0.02,
            "yaw_saturation_fraction": 0.0,
        }
    if risk_name == "moderate":
        return {
            "ram_mismatch": 0.16,
            "ram_uncertainty": 0.17,
            "recent_max_yaw_rate": 0.20,
            "recent_slip_score": 0.12,
            "body_stability_score": 0.86,
            "yaw_p95_abs_rate": 0.18,
            "yaw_mean_abs_rate": 0.06,
            "yaw_saturation_fraction": 0.03,
        }
    if risk_name == "high":
        return {
            "ram_mismatch": 0.45,
            "ram_uncertainty": 0.45,
            "recent_max_yaw_rate": 0.30,
            "recent_slip_score": 0.35,
            "body_stability_score": 0.72,
            "yaw_p95_abs_rate": 0.29,
            "yaw_mean_abs_rate": 0.12,
            "yaw_saturation_fraction": 0.25,
        }
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hc-sweep-root", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    args = ap.parse_args()

    root = Path(args.hc_sweep_root)
    csv_path = root / "hc_sweep_summary.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))

    examples = []

    for i, r in enumerate(rows):
        summary = load_json(r["summary_json"])

        risk_name = r["risk_name"]
        risk = risk_template(risk_name)

        action = {
            "vx_far": fget(r, "vx_far"),
            "vx_near": fget(r, "vx_near"),
            "goal_slow_distance": fget(r, "slow_distance"),
            "body_height": fget(r, "body_height"),
            "swing_clearance": fget(r, "swing_clearance"),
        }

        outcome = {
            "reached_stop_distance": bget(summary, "reached_stop_distance"),
            "min_rel_dist": fget(summary, "min_rel_dist"),
            "final_rel_dist": fget(summary, "final_rel_dist"),
            "progress_initial_minus_min": fget(summary, "progress_initial_minus_min"),
            "odom_x_delta": fget(summary, "odom_x_delta"),
            "max_mpc_vx": fget(summary, "max_mpc_vx"),
            "max_abs_mpc_yaw_rate": fget(summary, "max_abs_mpc_yaw_rate"),
            "duration_sec": fget(summary, "duration_sec"),
        }

        ex = {
            "episode_index": i + 1,
            "case_name": r["case_name"],
            "dataset_type": "phase_b_theta_hc_sweep_v0",
            "hc_sweep_root": str(root),
            "obs": {
                "risk_state": risk,
                "task": {
                    "world_name": summary.get("world_name", "earth"),
                    "model_name": summary.get("model_name", ""),
                    "goal_distance_ahead": fget(summary, "goal_distance_ahead", 0.5),
                    "goal_stop_distance": fget(summary, "goal_stop_distance", 0.15),
                },
            },
            "action": action,
            "outcome": outcome,
            "outcome_score": outcome_score(summary),
            "paths": {
                "summary_json": r["summary_json"],
                "run_dir": r["run_dir"],
                "csv_path": summary.get("csv_path", ""),
            },
        }

        examples.append(ex)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_jsonl, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, sort_keys=True) + "\n")

    by_risk = {}
    best_by_risk = {}

    for ex in examples:
        risk_name = ex["case_name"].split("_")[0]
        by_risk.setdefault(risk_name, 0)
        by_risk[risk_name] += 1

        current = best_by_risk.get(risk_name)
        if current is None or ex["outcome_score"] > current["outcome_score"]:
            best_by_risk[risk_name] = {
                "case_name": ex["case_name"],
                "outcome_score": ex["outcome_score"],
                "action": ex["action"],
                "outcome": ex["outcome"],
            }

    summary_out = {
        "hc_sweep_root": str(root),
        "num_examples": len(examples),
        "out_jsonl": args.out_jsonl,
        "by_risk": by_risk,
        "best_by_risk": best_by_risk,
    }

    with open(args.out_summary_json, "w") as f:
        json.dump(summary_out, f, indent=2, sort_keys=True)

    print(json.dumps(summary_out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
