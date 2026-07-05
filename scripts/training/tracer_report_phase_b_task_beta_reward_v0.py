#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def minmax(v, lo, hi):
    if hi <= lo:
        return 0.5
    return (v - lo) / (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta-report", default="reports/phase_b_beta_weighted_reward_v2_surrogate.json")
    ap.add_argument("--task-weight", type=float, default=0.65)
    ap.add_argument("--beta-weight", type=float, default=0.35)
    ap.add_argument("--out-json", default="reports/phase_b_task_beta_reward_v0.json")
    ap.add_argument("--out-md", default="reports/phase_b_task_beta_reward_v0.md")
    args = ap.parse_args()

    report = load_json(args.beta_report)
    summary = report["summary"]

    by_world = defaultdict(list)
    for r in summary:
        world = r["world_action"].split("::", 1)[0]
        by_world[world].append(r)

    rows = []

    for world, arr in sorted(by_world.items()):
        proxy_vals = [float(r["mean_proxy_reward"]) for r in arr]
        beta_vals = [float(r["mean_score_beta"]) for r in arr]
        reach_vals = [float(r["reach_rate"]) for r in arr]

        proxy_lo, proxy_hi = min(proxy_vals), max(proxy_vals)
        beta_lo, beta_hi = min(beta_vals), max(beta_vals)
        reach_lo, reach_hi = min(reach_vals), max(reach_vals)

        for r in arr:
            proxy = float(r["mean_proxy_reward"])
            beta = float(r["mean_score_beta"])
            reach = float(r["reach_rate"])

            proxy_norm = minmax(proxy, proxy_lo, proxy_hi)
            beta_norm = minmax(beta, beta_lo, beta_hi)
            reach_norm = minmax(reach, reach_lo, reach_hi)

            # Keep reach explicitly visible. Proxy already includes task quality,
            # but reach_norm prevents stable non-traversal from dominating.
            task_norm = 0.75 * proxy_norm + 0.25 * reach_norm

            score_task_beta = (
                args.task_weight * task_norm
                + args.beta_weight * beta_norm
            )

            row = dict(r)
            row.update({
                "world": world,
                "proxy_norm": proxy_norm,
                "beta_score_norm": beta_norm,
                "reach_norm": reach_norm,
                "task_norm": task_norm,
                "score_task_beta": score_task_beta,
            })
            rows.append(row)

    rows = sorted(rows, key=lambda r: (r["world"], -r["score_task_beta"]))

    out = {
        "schema": "phase_b_task_beta_reward_report_v0",
        "beta_report": args.beta_report,
        "task_weight": args.task_weight,
        "beta_weight": args.beta_weight,
        "summary": rows,
        "note": "Task-aware β reward. R_task remains mandatory; β-shaped reward adjusts the trade-off.",
    }
    save_json(Path(args.out_json), out)

    lines = []
    lines.append("# Phase-B Task-aware β Reward Report v0")
    lines.append("")
    lines.append(f"- Beta report: `{args.beta_report}`")
    lines.append(f"- Task weight: `{args.task_weight}`")
    lines.append(f"- Beta weight: `{args.beta_weight}`")
    lines.append("")
    lines.append("`score_task_beta = task_weight * task_norm + beta_weight * beta_score_norm`")
    lines.append("")
    lines.append("| world::action | n | reach | proxy | beta_score | task_norm | beta_norm | score_task_beta | final | drift |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        lines.append(
            f"| {r['world_action']} | {r['n']} | {r['reach_rate']:.3f} | "
            f"{r['mean_proxy_reward']:.3f} | {r['mean_score_beta']:.3f} | "
            f"{r['task_norm']:.3f} | {r['beta_score_norm']:.3f} | "
            f"{r['score_task_beta']:.3f} | {r['mean_final_dist']:.3f} | {r['mean_drift']:.3f} |"
        )

    md = "\n".join(lines) + "\n"
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md)

    print(md)
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
