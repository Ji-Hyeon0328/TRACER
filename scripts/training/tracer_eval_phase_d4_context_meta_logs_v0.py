#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path


PAT = re.compile(
    r"seq=(?P<seq>\d+).*?"
    r"context=(?P<context>\S+).*?"
    r"x=(?P<x>-?\d+\.\d+).*?"
    r"y=(?P<y>-?\d+\.\d+).*?"
    r"cmd=\[seq=(?P<cmd_seq>\d+), vx=(?P<vx>-?\d+\.\d+), yaw=(?P<yaw>-?\d+\.\d+), "
    r"h=(?P<h>-?\d+\.\d+), clr=(?P<clr>-?\d+\.\d+), enable=(?P<enable>-?\d+\.\d+)\].*?"
    r"out_of_lane=(?P<out_of_lane>\S+).*?"
    r"startup_failed=(?P<startup_failed>\S+).*?"
    r"stopped=(?P<stopped>\S+)"
)


def parse_log(path: Path, goal_x: float, lateral_bound: float):
    rows = []
    if not path.exists():
        return {
            "n": 0,
            "goal": False,
            "success": False,
            "startup_failed": True,
            "out_lane": False,
            "final_x": None,
            "final_y": None,
            "max_x": None,
            "max_abs_y": None,
            "mean_abs_y": None,
            "hold_drift": None,
            "first_goal_seq": None,
            "final_context": None,
        }

    for line in path.read_text(errors="replace").splitlines():
        m = PAT.search(line)
        if not m:
            continue
        d = m.groupdict()
        row = {
            "seq": int(d["seq"]),
            "context": d["context"],
            "x": float(d["x"]),
            "y": float(d["y"]),
            "vx": float(d["vx"]),
            "yaw": float(d["yaw"]),
            "h": float(d["h"]),
            "clr": float(d["clr"]),
            "enable": float(d["enable"]),
            "out_of_lane": d["out_of_lane"] == "True",
            "startup_failed": d["startup_failed"] == "True",
            "stopped": d["stopped"] == "True",
        }
        rows.append(row)

    if not rows:
        return {
            "n": 0,
            "goal": False,
            "success": False,
            "startup_failed": True,
            "out_lane": False,
            "final_x": None,
            "final_y": None,
            "max_x": None,
            "max_abs_y": None,
            "mean_abs_y": None,
            "hold_drift": None,
            "first_goal_seq": None,
            "final_context": None,
        }

    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    goal_rows = [r for r in rows if r["x"] >= goal_x]
    goal = bool(goal_rows)
    first_goal_seq = goal_rows[0]["seq"] if goal_rows else None
    max_x = max(xs)
    final_x = rows[-1]["x"]
    final_y = rows[-1]["y"]
    out_lane = any(abs(r["y"]) > lateral_bound or r["out_of_lane"] for r in rows)
    startup_failed = any(r["startup_failed"] for r in rows)

    # success = reached goal at least once and did not trigger safety failure.
    success = goal and (not out_lane) and (not startup_failed)

    # hold drift: how much x was lost after the maximum reached x.
    hold_drift = max_x - final_x if goal else None

    return {
        "n": len(rows),
        "goal": goal,
        "success": success,
        "startup_failed": startup_failed,
        "out_lane": out_lane,
        "final_x": final_x,
        "final_y": final_y,
        "max_x": max_x,
        "max_abs_y": max(abs(y) for y in ys),
        "mean_abs_y": sum(abs(y) for y in ys) / len(ys),
        "hold_drift": hold_drift,
        "first_goal_seq": first_goal_seq,
        "final_context": rows[-1]["context"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rollout_rows = []
    with manifest.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            log_dir = Path(row["log_dir"])
            goal_x = float(row["goal_x"])
            lateral = float(row["lateral_bound"])
            metrics = parse_log(log_dir / "context_meta_selector.log", goal_x, lateral)
            rollout_rows.append({**row, **metrics})

    n = len(rollout_rows)
    success_rate = sum(1 for r in rollout_rows if r["success"]) / n if n else 0.0
    goal_rate = sum(1 for r in rollout_rows if r["goal"]) / n if n else 0.0
    startup_failed_rate = sum(1 for r in rollout_rows if r["startup_failed"]) / n if n else 0.0
    out_lane_rate = sum(1 for r in rollout_rows if r["out_lane"]) / n if n else 0.0

    hold_vals = [r["hold_drift"] for r in rollout_rows if r["hold_drift"] is not None]
    hold_drift_mean = sum(hold_vals) / len(hold_vals) if hold_vals else None

    summary = {
        "manifest": str(manifest),
        "n": n,
        "success_rate": success_rate,
        "goal_rate": goal_rate,
        "startup_failed_rate": startup_failed_rate,
        "out_lane_rate": out_lane_rate,
        "hold_drift_mean": hold_drift_mean,
        "rollouts": rollout_rows,
    }

    base = str(out_prefix)
    csv_path = Path(base + "_rollouts_v0.csv")
    json_path = Path(base + "_summary_v0.json")
    md_path = Path(base + "_summary_v0.md")

    fieldnames = [
        "label", "trial", "world", "goal_x", "lateral_bound", "hold_vx", "log_dir",
        "success", "goal", "startup_failed", "out_lane",
        "final_x", "final_y", "max_x", "max_abs_y", "mean_abs_y",
        "hold_drift", "first_goal_seq", "final_context",
    ]

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rollout_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    json_path.write_text(json.dumps(summary, indent=2))

    lines = []
    lines.append("# TRACER Phase-D4 Context-Meta Repeat Summary v0")
    lines.append("")
    lines.append(f"- manifest: `{manifest}`")
    lines.append(f"- n: `{n}`")
    lines.append(f"- success_rate: `{success_rate:.3f}`")
    lines.append(f"- goal_rate: `{goal_rate:.3f}`")
    lines.append(f"- startup_failed_rate: `{startup_failed_rate:.3f}`")
    lines.append(f"- out_lane_rate: `{out_lane_rate:.3f}`")
    lines.append(f"- hold_drift_mean: `{hold_drift_mean if hold_drift_mean is not None else 'NA'}`")
    lines.append("")
    lines.append("## Individual rollouts")
    lines.append("")
    lines.append("| trial | success | goal | startup_failed | out_lane | final_x | final_y | max_x | max_abs_y | mean_abs_y | hold_drift | final_context |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rollout_rows:
        def fmt(v):
            if v is None:
                return "NA"
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)
        lines.append(
            f"| {r['trial']} | {r.get('hold_vx', 'NA')} | {r['success']} | {r['goal']} | {r['startup_failed']} | {r['out_lane']} | "
            f"{fmt(r['final_x'])} | {fmt(r['final_y'])} | {fmt(r['max_x'])} | "
            f"{fmt(r['max_abs_y'])} | {fmt(r['mean_abs_y'])} | {fmt(r['hold_drift'])} | {r['final_context']} |"
        )

    md_path.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] wrote {csv_path}")
    print(f"[TRACER] wrote {json_path}")
    print(f"[TRACER] wrote {md_path}")


if __name__ == "__main__":
    main()
