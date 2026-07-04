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
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.result_root).rglob("ppo_policy_episode_result_v0.json")):
        r = load_json(p)
        sel = r.get("learned_score_selector", {})
        c = r.get("reward_components", {})
        top = sel.get("ranked_actions", [])[:5]

        rows.append({
            "path": str(p),
            "world_name": r.get("world_name"),
            "raw_action": sel.get("raw_argmax_action"),
            "selected_action": r.get("selected_action_name"),
            "changed": sel.get("changed_action"),
            "alpha": r.get("alpha"),
            "reached": c.get("reached_stop_distance"),
            "reach_reward": c.get("reach_reward"),
            "hold_reward": c.get("hold_reward"),
            "min_rel_dist": c.get("min_rel_dist"),
            "final_rel_dist": c.get("final_rel_dist"),
            "top5": [
                {
                    "action": a.get("action_name"),
                    "prob": a.get("prob"),
                    "learned_z": a.get("learned_score_z"),
                    "final_score": a.get("final_score"),
                    "has_learned_score": a.get("has_learned_score")
                }
                for a in top
            ]
        })

    report = {
        "schema": "phase_b_learned_score_selector_decision_report_v0",
        "result_root": args.result_root,
        "rows": rows,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = []
    lines.append("# Phase-B Learned Score Selector Decisions")
    lines.append("")
    lines.append("| world | raw | selected | changed | reached | reach | hold | final_dist | top3 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")

    for r in rows:
        top3 = ", ".join(
            f"{a['action']}({fmt(a['final_score'])})"
            for a in r["top5"][:3]
        )
        lines.append(
            f"| {r['world_name']} | {r['raw_action']} | {r['selected_action']} | "
            f"{r['changed']} | {r['reached']} | {fmt(r['reach_reward'])} | "
            f"{fmt(r['hold_reward'])} | {fmt(r['final_rel_dist'])} | {top3} |"
        )

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
