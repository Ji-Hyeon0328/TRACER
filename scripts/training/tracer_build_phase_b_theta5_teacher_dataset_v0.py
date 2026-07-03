#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def to_float(x, default=0.0):
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def to_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in ("true", "1", "yes")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def score_row(row):
    reached = to_bool(row.get("reached"))
    progress = to_float(row.get("progress"))
    odom = to_float(row.get("odom_x_delta"))
    min_dist = to_float(row.get("min_rel_dist"), 999.0)
    yaw = to_float(row.get("max_yaw"), 999.0)
    vx = to_float(row.get("vx_far"))

    score = 0.0
    score += 100.0 if reached else 0.0
    score += 40.0 * max(progress, 0.0)
    score += 8.0 * max(odom, 0.0)
    score += 3.0 * max(vx, 0.0)
    score -= 20.0 * max(min_dist, 0.0)
    score -= 15.0 * max(yaw - 0.18, 0.0)
    score -= 30.0 * max(yaw - 0.28, 0.0)
    return score


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def infer_world(csv_path, row):
    summary_json = row.get("summary_json", "")
    if summary_json and Path(summary_json).is_file():
        try:
            s = load_json(summary_json)
            if s.get("world_name"):
                return str(s["world_name"])
        except Exception:
            pass

    parent = Path(csv_path).parent.name
    if parent:
        return parent
    return "unknown"


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


def summarize_case(rows):
    scores = [to_float(r["_score"]) for r in rows]
    reached = [to_bool(r.get("reached")) for r in rows]

    def avg_field(k):
        return mean([to_float(r.get(k)) for r in rows])

    return {
        "n": len(rows),
        "score_mean": mean(scores),
        "score_std": stdev(scores),
        "reached_rate": sum(1 for x in reached if x) / max(len(reached), 1),
        "action": {
            "vx_far": avg_field("vx_far"),
            "vx_near": avg_field("vx_near"),
            "goal_slow_distance": avg_field("slow_distance"),
            "body_height": avg_field("body_height"),
            "swing_clearance": avg_field("swing_clearance"),
        },
        "outcome": {
            "min_rel_dist": avg_field("min_rel_dist"),
            "final_rel_dist": avg_field("final_rel_dist"),
            "progress_initial_minus_min": avg_field("progress"),
            "odom_x_delta": avg_field("odom_x_delta"),
            "max_mpc_vx": avg_field("max_vx"),
            "max_abs_mpc_yaw_rate": avg_field("max_yaw"),
            "reached_stop_distance_rate": sum(1 for x in reached if x) / max(len(reached), 1),
        },
        "source_csvs": sorted(set(r["_source_csv"] for r in rows)),
        "source_summaries": sorted(set(r.get("summary_json", "") for r in rows if r.get("summary_json", ""))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", nargs="+", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    args = ap.parse_args()

    by_world_risk_case = defaultdict(list)

    for csv_s in args.summary_csv:
        csv_path = Path(csv_s)
        if not csv_path.is_file():
            print(f"[WARN] missing csv: {csv_path}")
            continue

        with open(csv_path, "r") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            world = infer_world(csv_path, row)
            risk = row.get("risk_name", "")
            case = row.get("case_name", "")

            item = dict(row)
            item["_world"] = world
            item["_risk"] = risk
            item["_case"] = case
            item["_source_csv"] = str(csv_path)
            item["_score"] = score_row(row)

            by_world_risk_case[(world, risk, case)].append(item)

    case_summaries = {}
    for key, rows in by_world_risk_case.items():
        case_summaries[key] = summarize_case(rows)

    by_world_risk = defaultdict(list)
    for (world, risk, case), summary in case_summaries.items():
        by_world_risk[(world, risk)].append((case, summary))

    examples = []
    best_table = {}

    for (world, risk), candidates in sorted(by_world_risk.items()):
        candidates.sort(
            key=lambda x: (
                x[1]["score_mean"],
                x[1]["reached_rate"],
                -x[1]["score_std"],
            ),
            reverse=True,
        )

        best_case, best = candidates[0]
        action = best["action"]

        ex = {
            "dataset_type": "phase_b_theta5_teacher_v0",
            "world_name": world,
            "risk_name": risk,
            "teacher_case_name": best_case,
            "obs": {
                "risk_state": risk_template(risk),
                "task": {
                    "world_name": world,
                    "goal_distance_ahead": 0.5,
                    "goal_stop_distance": 0.15,
                },
            },
            "action": action,
            "teacher_stats": {
                "n": best["n"],
                "score_mean": best["score_mean"],
                "score_std": best["score_std"],
                "reached_rate": best["reached_rate"],
                "outcome": best["outcome"],
                "source_csvs": best["source_csvs"],
                "source_summaries": best["source_summaries"],
            },
            "all_candidate_cases": [
                {
                    "case_name": case,
                    "score_mean": s["score_mean"],
                    "score_std": s["score_std"],
                    "reached_rate": s["reached_rate"],
                    "action": s["action"],
                    "outcome": s["outcome"],
                    "n": s["n"],
                }
                for case, s in candidates
            ],
        }

        examples.append(ex)
        best_table[f"{world}::{risk}"] = {
            "case_name": best_case,
            "action": action,
            "score_mean": best["score_mean"],
            "score_std": best["score_std"],
            "reached_rate": best["reached_rate"],
            "n": best["n"],
        }

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_jsonl, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, sort_keys=True) + "\n")

    summary = {
        "num_examples": len(examples),
        "num_case_groups": len(case_summaries),
        "out_jsonl": args.out_jsonl,
        "best_table": best_table,
    }

    with open(args.out_summary_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
