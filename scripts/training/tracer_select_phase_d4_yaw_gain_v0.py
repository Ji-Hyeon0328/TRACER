#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-manifest", required=True)
    ap.add_argument("--goal-x", type=float, default=8.0)
    ap.add_argument("--pub-hz", type=float, default=10.0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    bank_manifest = Path(args.bank_manifest)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with bank_manifest.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            summary = json.loads(Path(row["summary_json"]).read_text())
            rollouts = summary["rollouts"]

            final_goal_errors = [
                abs(float(r["final_x"]) - args.goal_x)
                for r in rollouts if r["final_x"] is not None
            ]
            hold_drifts = [
                float(r["hold_drift"])
                for r in rollouts if r["hold_drift"] is not None
            ]
            max_abs_ys = [
                float(r["max_abs_y"])
                for r in rollouts if r["max_abs_y"] is not None
            ]
            mean_abs_ys = [
                float(r["mean_abs_y"])
                for r in rollouts if r["mean_abs_y"] is not None
            ]
            first_goal_times = [
                float(r["first_goal_seq"]) / args.pub_hz
                for r in rollouts if r.get("first_goal_seq") is not None
            ]

            success_rate = float(summary["success_rate"])
            goal_rate = float(summary["goal_rate"])
            startup_failed_rate = float(summary["startup_failed_rate"])
            out_lane_rate = float(summary["out_lane_rate"])

            final_goal_error_mean = mean(final_goal_errors)
            hold_drift_mean = mean(hold_drifts)
            max_abs_y_mean = mean(max_abs_ys)
            mean_abs_y_mean = mean(mean_abs_ys)
            first_goal_time_mean = mean(first_goal_times)

            # Higher is better.
            # For D4.4, lateral metrics are weighted strongly.
            score = (
                100.0 * success_rate
                + 20.0 * goal_rate
                - 0.08 * (first_goal_time_mean if first_goal_time_mean is not None else 260.0)
                - 18.0 * (final_goal_error_mean if final_goal_error_mean is not None else 10.0)
                - 12.0 * (hold_drift_mean if hold_drift_mean is not None else 10.0)
                - 35.0 * (max_abs_y_mean if max_abs_y_mean is not None else 10.0)
                - 18.0 * (mean_abs_y_mean if mean_abs_y_mean is not None else 10.0)
                - 100.0 * startup_failed_rate
                - 100.0 * out_lane_rate
            )

            rows.append({
                "yaw_gain": float(row["yaw_gain"]),
                "yaw_sign": float(row["yaw_sign"]),
                "yaw_k": float(row["yaw_k"]),
                "yaw_max": float(row["yaw_max"]),
                "score": score,
                "success_rate": success_rate,
                "goal_rate": goal_rate,
                "startup_failed_rate": startup_failed_rate,
                "out_lane_rate": out_lane_rate,
                "first_goal_time_mean": first_goal_time_mean,
                "final_goal_error_mean": final_goal_error_mean,
                "hold_drift_mean": hold_drift_mean,
                "max_abs_y_mean": max_abs_y_mean,
                "mean_abs_y_mean": mean_abs_y_mean,
                "policy_json": row["policy_json"],
                "summary_json": row["summary_json"],
                "summary_md": row["summary_md"],
            })

    rows.sort(key=lambda r: r["score"], reverse=True)
    best = rows[0] if rows else None

    csv_path = Path(str(out_prefix) + "_comparison_v0.csv")
    json_path = Path(str(out_prefix) + "_best_v0.json")
    md_path = Path(str(out_prefix) + "_comparison_v0.md")

    fields = [
        "yaw_gain", "yaw_sign", "yaw_k", "yaw_max", "score",
        "success_rate", "goal_rate", "startup_failed_rate", "out_lane_rate",
        "first_goal_time_mean", "final_goal_error_mean", "hold_drift_mean",
        "max_abs_y_mean", "mean_abs_y_mean", "policy_json", "summary_json", "summary_md",
    ]

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    json_path.write_text(json.dumps({"best": best, "rows": rows}, indent=2))

    def fmt(v):
        if v is None:
            return "NA"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = []
    lines.append("# TRACER Phase-D4 Yaw Gain Bank Comparison v0")
    lines.append("")
    lines.append(f"- bank_manifest: `{bank_manifest}`")
    if best:
        lines.append(f"- best_yaw_gain: `{best['yaw_gain']:.4f}`")
        lines.append(f"- best_yaw_sign: `{best['yaw_sign']:.4f}`")
        lines.append(f"- best_yaw_k: `{best['yaw_k']:.4f}`")
        lines.append(f"- best_score: `{best['score']:.4f}`")
    lines.append("")
    lines.append("| rank | yaw_gain | score | success_rate | goal_rate | first_goal_time_mean | final_goal_error_mean | hold_drift_mean | max_abs_y_mean | mean_abs_y_mean |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {fmt(r['yaw_gain'])} | {fmt(r['score'])} | {fmt(r['success_rate'])} | "
            f"{fmt(r['goal_rate'])} | {fmt(r['first_goal_time_mean'])} | "
            f"{fmt(r['final_goal_error_mean'])} | {fmt(r['hold_drift_mean'])} | "
            f"{fmt(r['max_abs_y_mean'])} | {fmt(r['mean_abs_y_mean'])} |"
        )

    md_path.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] wrote {csv_path}")
    print(f"[TRACER] wrote {json_path}")
    print(f"[TRACER] wrote {md_path}")
    if best:
        print(f"[TRACER] best yaw_gain={best['yaw_gain']:.4f}, score={best['score']:.4f}")


if __name__ == "__main__":
    main()
