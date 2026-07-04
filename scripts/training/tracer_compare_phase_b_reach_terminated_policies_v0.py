#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def fmt(x):
    if x is None:
        return ""
    return f"{float(x):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("reports", nargs="+")
    args = ap.parse_args()

    rows = []
    for rp in args.reports:
        r = load_json(rp)
        policy = r.get("policy_name", Path(rp).stem)
        for world, g in r.get("by_world", {}).items():
            rows.append({
                "policy": policy,
                "world": world,
                "n": g.get("n"),
                "reach_rate": g.get("reach_rate"),
                "mean_min_rel_dist": g.get("mean_min_rel_dist"),
                "mean_final_rel_dist": g.get("mean_final_rel_dist"),
                "mean_post_reach_drift": g.get("mean_post_reach_drift"),
                "mean_R_v": g.get("mean_R_v"),
                "mean_R_s": g.get("mean_R_s"),
                "mean_reward": g.get("mean_tracer_proxy_reward"),
            })

    out = {
        "schema": "phase_b_reach_terminated_policy_comparison_v0",
        "reports": args.reports,
        "rows": rows,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True))

    lines = []
    lines.append("# Phase-B Reach-Terminated Policy Comparison")
    lines.append("")
    lines.append("Main metric: reach-terminated traversal success. Final distance and R_s are auxiliary frozen-controller stability diagnostics.")
    lines.append("")
    lines.append("| world | policy | n | reach_rate | mean_min_dist | mean_final_dist | post_reach_drift | mean_R_v | mean_R_s | mean_reward |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in sorted(rows, key=lambda x: (x["world"], x["policy"])):
        lines.append(
            f"| {row['world']} | {row['policy']} | {row['n']} | "
            f"{fmt(row['reach_rate'])} | {fmt(row['mean_min_rel_dist'])} | "
            f"{fmt(row['mean_final_rel_dist'])} | {fmt(row['mean_post_reach_drift'])} | "
            f"{fmt(row['mean_R_v'])} | {fmt(row['mean_R_s'])} | {fmt(row['mean_reward'])} |"
        )

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
