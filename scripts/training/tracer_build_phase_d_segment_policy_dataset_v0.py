#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path
from collections import defaultdict


PAT = re.compile(
    r"seq=(?P<seq>\d+).*?x=(?P<x>[-+0-9.]+).*?y=(?P<y>[-+0-9.]+).*?"
    r"seg=(?P<seg>[^ ]+).*?cmd=\[vx=(?P<vx>[-+0-9.]+), yaw=(?P<yaw>[-+0-9.]+), "
    r"h=(?P<h>[-+0-9.]+), clr=(?P<clr>[-+0-9.]+)\].*?stopped=(?P<stopped>True|False)"
)


def parse_log(log_path):
    rows = []
    if not log_path.exists():
        return rows
    for line in log_path.read_text(errors="ignore").splitlines():
        m = PAT.search(line)
        if not m:
            continue
        d = m.groupdict()
        rows.append({
            "seq": int(d["seq"]),
            "x": float(d["x"]),
            "y": float(d["y"]),
            "seg": d["seg"],
            "vx": float(d["vx"]),
            "h": float(d["h"]),
            "clr": float(d["clr"]),
            "stopped": d["stopped"] == "True",
        })
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--goal-x", type=float, default=8.0)
    ap.add_argument("--lateral-bound", type=float, default=2.0)
    ap.add_argument("--timeout-s", type=float, default=460.0)
    ap.add_argument("--pub-hz", type=float, default=10.0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rollouts = []

    with manifest.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            log_dir = Path(row["log_dir"])
            samples = parse_log(log_dir / "segment_action.log")

            if not samples:
                final_x = 0.0
                final_y = 0.0
                max_abs_y = 0.0
                mean_abs_y = 0.0
                goal = False
                stopped = False
                time_s = None
                startup_failed = True
            else:
                final_x = samples[-1]["x"]
                final_y = samples[-1]["y"]
                max_abs_y = max(abs(s["y"]) for s in samples)
                mean_abs_y = mean([abs(s["y"]) for s in samples])
                goal_idx = next((i for i, s in enumerate(samples) if s["x"] >= args.goal_x), None)
                goal = goal_idx is not None
                stopped = any(s["stopped"] for s in samples)
                time_s = samples[goal_idx]["seq"] / args.pub_hz if goal else None
                startup_failed = len(samples) >= 20 and max(abs(s["x"]) for s in samples) < 0.20

            out_lane = max_abs_y > args.lateral_bound
            success = bool(goal and not out_lane and not startup_failed)

            progress = max(0.0, min(final_x / args.goal_x, 1.0))
            time_cost = time_s if time_s is not None else args.timeout_s
            score = (
                100.0 * float(success)
                + 30.0 * progress
                - 80.0 * float(out_lane)
                - 10.0 * max_abs_y
                - 5.0 * mean_abs_y
                - 0.03 * time_cost
            )

            rollouts.append({
                "label": row["label"],
                "policy": row.get("policy", "segment"),
                "trial": int(row["trial"]),
                "schedule": row["schedule"],
                "success": success,
                "goal": goal,
                "out_lane": out_lane,
                "startup_failed": startup_failed,
                "stopped": stopped,
                "time": time_s,
                "max_abs_y": max_abs_y,
                "mean_abs_y": mean_abs_y,
                "final_x": final_x,
                "final_y": final_y,
                "score": score,
                "log_dir": str(log_dir),
            })

    rollout_csv = out_prefix.with_name(out_prefix.name + "_rollouts_v0.csv")
    with rollout_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rollouts[0].keys()))
        writer.writeheader()
        writer.writerows(rollouts)

    groups = defaultdict(list)
    for r in rollouts:
        groups[r["label"]].append(r)

    ranking = []
    for label, rs in groups.items():
        ranking.append({
            "label": label,
            "n": len(rs),
            "schedule": rs[0]["schedule"],
            "success_rate": mean([float(r["success"]) for r in rs]),
            "goal_rate": mean([float(r["goal"]) for r in rs]),
            "out_lane_rate": mean([float(r["out_lane"]) for r in rs]),
            "startup_failed_rate": mean([float(r["startup_failed"]) for r in rs]),
            "score_mean": mean([r["score"] for r in rs]),
            "score_std": std([r["score"] for r in rs]),
            "time_mean": mean([r["time"] for r in rs]),
            "max_abs_y_mean": mean([r["max_abs_y"] for r in rs]),
            "mean_abs_y_mean": mean([r["mean_abs_y"] for r in rs]),
        })

    ranking.sort(key=lambda x: x["score_mean"], reverse=True)

    summary = {
        "manifest": str(manifest),
        "goal_x": args.goal_x,
        "lateral_bound": args.lateral_bound,
        "rollout_csv": str(rollout_csv),
        "best_label": ranking[0]["label"] if ranking else None,
        "ranking": ranking,
    }

    out_json = out_prefix.with_name(out_prefix.name + "_summary_v0.json")
    out_json.write_text(json.dumps(summary, indent=2))

    out_md = out_prefix.with_name(out_prefix.name + "_summary_v0.md")
    lines = []
    lines.append("# TRACER Phase-D Segment Policy Dataset Summary v0\n")
    lines.append(f"- manifest: `{manifest}`")
    lines.append(f"- goal_x: `{args.goal_x}`")
    lines.append(f"- lateral_bound: `{args.lateral_bound}`")
    lines.append(f"- rollout_csv: `{rollout_csv}`")
    lines.append(f"- best_label: `{summary['best_label']}`")
    lines.append("\n## Group ranking\n")
    lines.append("| rank | label | n | success_rate | goal_rate | out_lane_rate | startup_failed_rate | score_mean | score_std | time_mean | max_abs_y_mean | mean_abs_y_mean |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(ranking, 1):
        lines.append(
            f"| {i} | {r['label']} | {r['n']} | {fmt(r['success_rate'])} | {fmt(r['goal_rate'])} | "
            f"{fmt(r['out_lane_rate'])} | {fmt(r['startup_failed_rate'])} | {fmt(r['score_mean'])} | "
            f"{fmt(r['score_std'])} | {fmt(r['time_mean'])} | {fmt(r['max_abs_y_mean'])} | {fmt(r['mean_abs_y_mean'])} |"
        )

    lines.append("\n## Policies\n")
    lines.append("| label | schedule |")
    lines.append("|---|---|")
    for r in ranking:
        lines.append(f"| {r['label']} | `{r['schedule']}` |")

    lines.append("\n## Individual rollouts\n")
    lines.append("| label | trial | success | goal | startup_failed | time | max_abs_y | mean_abs_y | final_x | final_y | score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rollouts:
        lines.append(
            f"| {r['label']} | {r['trial']} | {r['success']} | {r['goal']} | {r['startup_failed']} | "
            f"{fmt(r['time'])} | {fmt(r['max_abs_y'])} | {fmt(r['mean_abs_y'])} | "
            f"{fmt(r['final_x'])} | {fmt(r['final_y'])} | {fmt(r['score'])} |"
        )

    out_md.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] wrote {rollout_csv}")
    print(f"[TRACER] wrote {out_json}")
    print(f"[TRACER] wrote {out_md}")


if __name__ == "__main__":
    main()
