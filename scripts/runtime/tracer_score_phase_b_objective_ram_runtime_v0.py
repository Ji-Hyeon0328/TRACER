#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import math
from pathlib import Path


def ff(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def bb(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() == "true"


def actual_labels(summary):
    stop = ff(summary.get("goal_stop_distance"), 0.15)
    reached = bb(summary.get("reached_stop_distance"))
    min_d = ff(summary.get("min_rel_dist"), 999.0)
    final = ff(summary.get("final_rel_dist"), 999.0)
    dx = ff(summary.get("odom_x_delta"), 0.0)
    yaw = ff(summary.get("max_abs_mpc_yaw_rate"), 0.0)
    progress = ff(summary.get("progress_initial_minus_min"), 0.0)

    approach = min_d <= stop
    stable = reached and final <= 0.30 and abs(dx) <= 1.00
    drift = approach and final > 0.30
    yaw_sat = yaw >= 0.299
    future_invalid = (not stable) and (final > 0.50 or abs(dx) > 1.00)
    recovery_needed = (not stable) or drift or yaw_sat or future_invalid

    return {
        "actual_reached": reached,
        "actual_approach_success": approach,
        "actual_stable_reached": stable,
        "actual_drift_after_approach": drift,
        "actual_yaw_saturated": yaw_sat,
        "actual_future_invalid": future_invalid,
        "actual_recovery_needed": recovery_needed,
        "actual_min_rel_dist": min_d,
        "actual_final_rel_dist": final,
        "actual_odom_x_delta": dx,
        "actual_abs_odom_x_delta": abs(dx),
        "actual_max_abs_yaw_rate": yaw,
        "actual_progress": progress,
    }


def actual_semantic(labels):
    if labels["actual_stable_reached"]:
        return "stable_goal_reach_flat_locomotion"
    if labels["actual_approach_success"] and labels["actual_drift_after_approach"]:
        return "approach_possible_but_post_reach_hold_needed"
    if labels["actual_final_rel_dist"] > 0.75 or labels["actual_abs_odom_x_delta"] > 1.0:
        return "forward_walk_unreliable_on_soft_terrain"
    return "cautious_probe_required"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-glob", default="artifacts/phase_b_theta5_predicted_goal_episode_*")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    run_dirs = sorted(glob.glob(args.run_glob), key=lambda p: Path(p).stat().st_mtime, reverse=True)

    rows = []

    for d in run_dirs:
        summary_path = Path(d) / "episode" / "phase_b_summary.json"
        pred_path = Path(d) / "theta5_prediction.json"

        if not summary_path.is_file():
            continue

        with open(summary_path, "r") as f:
            s = json.load(f)

        labels = actual_labels(s)
        actual_sem = actual_semantic(labels)

        pred_sem = s.get("objective_semantic")
        pred_risk = ff(s.get("predicted_future_risk"), 0.0)
        pred_recovery = s.get("predicted_recovery_needed")
        pred_invalid = s.get("predicted_future_invalid")

        runtime_decision = s.get("runtime_decision", {}) or {}

        pred_stable = runtime_decision.get("stable_reached_pred")
        pred_approach = runtime_decision.get("approach_success_pred")
        pred_drift = runtime_decision.get("drift_after_approach_pred")
        pred_yaw_sat = runtime_decision.get("yaw_saturated_pred")

        # Risk target from actual outcome, same spirit as previous bootstrap labels.
        actual_risk = 0.0
        actual_risk += 0.35 if not labels["actual_stable_reached"] else 0.0
        actual_risk += 0.20 if labels["actual_drift_after_approach"] else 0.0
        actual_risk += 0.20 if labels["actual_yaw_saturated"] else 0.0
        actual_risk += min(0.20, max(labels["actual_final_rel_dist"] - 0.30, 0.0) * 0.20)
        actual_risk += min(0.15, max(labels["actual_abs_odom_x_delta"] - 0.75, 0.0) * 0.10)
        actual_risk = max(0.0, min(1.0, actual_risk))

        row = {
            "run_dir": d,
            "summary_json": str(summary_path),
            "prediction_json": str(pred_path) if pred_path.is_file() else "",
            "world_name": s.get("world_name"),
            "theta5_profile_name": s.get("theta5_profile_name"),
            "selected_profile_name": s.get("selected_profile_name"),
            "vx_far": s.get("vx_far"),
            "vx_near": s.get("vx_near"),
            "body_height": s.get("body_height"),
            "swing_clearance": s.get("swing_clearance"),

            "pred_semantic": pred_sem,
            "actual_semantic": actual_sem,
            "semantic_match": pred_sem == actual_sem,

            "pred_future_risk": pred_risk,
            "actual_future_risk": actual_risk,
            "risk_error": pred_risk - actual_risk,

            "pred_recovery_needed": pred_recovery,
            "actual_recovery_needed": labels["actual_recovery_needed"],
            "recovery_match": pred_recovery == labels["actual_recovery_needed"],

            "pred_future_invalid": pred_invalid,
            "actual_future_invalid": labels["actual_future_invalid"],
            "future_invalid_match": pred_invalid == labels["actual_future_invalid"],

            "pred_stable_reached": pred_stable,
            "actual_stable_reached": labels["actual_stable_reached"],
            "stable_match": pred_stable == labels["actual_stable_reached"],

            "pred_approach_success": pred_approach,
            "actual_approach_success": labels["actual_approach_success"],
            "approach_match": pred_approach == labels["actual_approach_success"],

            "pred_drift_after_approach": pred_drift,
            "actual_drift_after_approach": labels["actual_drift_after_approach"],
            "drift_match": pred_drift == labels["actual_drift_after_approach"],

            "pred_yaw_saturated": pred_yaw_sat,
            "actual_yaw_saturated": labels["actual_yaw_saturated"],
            "yaw_sat_match": pred_yaw_sat == labels["actual_yaw_saturated"],

            **labels,
        }

        rows.append(row)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.out_dir) / "objective_ram_runtime_score_v0.csv"
    out_json = Path(args.out_dir) / "objective_ram_runtime_score_summary_v0.json"

    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(out_csv, "w") as f:
            f.write("")

    def rate(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return sum(1 for v in vals if v) / len(vals)

    risk_mae = None
    if rows:
        risk_mae = sum(abs(ff(r["risk_error"])) for r in rows) / len(rows)

    summary = {
        "num_runs": len(rows),
        "out_csv": str(out_csv),
        "semantic_match_rate": rate("semantic_match"),
        "recovery_match_rate": rate("recovery_match"),
        "future_invalid_match_rate": rate("future_invalid_match"),
        "stable_match_rate": rate("stable_match"),
        "approach_match_rate": rate("approach_match"),
        "drift_match_rate": rate("drift_match"),
        "yaw_sat_match_rate": rate("yaw_sat_match"),
        "future_risk_mae": risk_mae,
        "runs": rows,
    }

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
