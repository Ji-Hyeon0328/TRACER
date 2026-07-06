#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import defaultdict

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/phase_b_sponge_late_switch_v1d_eval")
    ap.add_argument("--out-jsonl", default="data/phase_b_invalid_gait_labels_v0/sponge_late_switch_v1d_suspicious_labels.jsonl")
    ap.add_argument("--out-report-json", default="reports/phase_b_invalid_gait_labels_v0_sponge_late_switch_v1d.json")
    ap.add_argument("--out-report-md", default="reports/phase_b_invalid_gait_labels_v0_sponge_late_switch_v1d.md")
    args = ap.parse_args()

    root = Path(args.root)
    summaries = sorted(root.rglob("phase_b_summary.json"))

    rows = []
    by_action = defaultdict(lambda: {"n": 0, "suspicious": 0, "reached": 0})

    for p in summaries:
        s = load_json(p)
        run_dir = Path(s.get("run_dir", p.parent))
        action = run_dir.parent.name
        world = s.get("world_name", "unknown")

        min_dist = float(s.get("min_rel_dist", 999.0))
        final_dist = float(s.get("final_rel_dist", 999.0))
        drift = max(0.0, final_dist - min_dist)
        reached = bool(s.get("reached_stop_distance", False))
        odom_delta = float(s.get("odom_x_delta", 0.0))
        progress_min = float(s.get("progress_initial_minus_min", 0.0))
        progress_final = float(s.get("progress_initial_minus_final", 0.0))

        # Weak suspicious criteria:
        # 1) Looks stable by final distance but rarely reaches.
        # 2) Very low progress or crawl/drag-like progress.
        # 3) Final is good but min_dist not close enough, suggesting early hold/drag.
        suspicious = False
        reasons = []

        if (not reached) and final_dist < 1.20:
            suspicious = True
            reasons.append("low_final_without_reach")

        if progress_min < 0.20:
            suspicious = True
            reasons.append("low_progress_to_goal")

        if final_dist < 1.20 and min_dist > 0.30:
            suspicious = True
            reasons.append("stable_but_not_close_to_goal")

        if abs(odom_delta) < 0.15:
            suspicious = True
            reasons.append("very_low_odom_delta")

        row = {
            "schema": "phase_b_suspicious_gait_label_v0",
            "source": "summary_weak_label",
            "world": world,
            "action": action,
            "episode_dir": str(p.parent),
            "summary_json": str(p),
            "reached": reached,
            "min_rel_dist": min_dist,
            "final_rel_dist": final_dist,
            "post_reach_drift_proxy": drift,
            "odom_x_delta": odom_delta,
            "progress_initial_minus_min": progress_min,
            "progress_initial_minus_final": progress_final,
            "suspicious_invalid_gait": suspicious,
            "reasons": reasons,
        }
        rows.append(row)

        by_action[action]["n"] += 1
        by_action[action]["suspicious"] += int(suspicious)
        by_action[action]["reached"] += int(reached)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    report = {
        "schema": "phase_b_suspicious_gait_label_report_v0",
        "root": args.root,
        "num_episodes": len(rows),
        "num_suspicious": sum(int(r["suspicious_invalid_gait"]) for r in rows),
        "by_action": {
            k: {
                **v,
                "suspicious_rate": v["suspicious"] / max(1, v["n"]),
                "reach_rate": v["reached"] / max(1, v["n"]),
            }
            for k, v in sorted(by_action.items())
        },
    }

    Path(args.out_report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report_json, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    lines = []
    lines.append("# Phase-B Suspicious Sponge Gait Labels v0")
    lines.append("")
    lines.append(f"- Root: `{args.root}`")
    lines.append(f"- Episodes: `{len(rows)}`")
    lines.append(f"- Suspicious: `{report['num_suspicious']}`")
    lines.append("")
    lines.append("## By action")
    lines.append("")
    lines.append("| action | n | reach_rate | suspicious_rate | suspicious |")
    lines.append("|---|---:|---:|---:|---:|")
    for action, v in report["by_action"].items():
        lines.append(
            f"| {action} | {v['n']} | {v['reach_rate']:.3f} | {v['suspicious_rate']:.3f} | {v['suspicious']} |"
        )

    md = "\n".join(lines) + "\n"
    with open(args.out_report_md, "w") as f:
        f.write(md)

    print(md)
    print("[wrote]", args.out_jsonl)

if __name__ == "__main__":
    main()
