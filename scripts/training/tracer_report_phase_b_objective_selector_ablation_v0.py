#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def fmt(x):
    if x is None:
        return ""
    return f"{float(x):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-json", default="reports/phase_b_reach_terminated_policy_comparison_objective_selector_v0.json")
    ap.add_argument("--raw-policy", default="tracer_proxy_ppo_argmax_v1")
    ap.add_argument("--selector-policy", default="tracer_ppo_with_objective_selector_ram_proxy_v0")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    comp = load_json(args.comparison_json)
    rows = comp.get("rows", [])

    by_world = {}
    for r in rows:
        by_world.setdefault(r["world"], {})[r["policy"]] = r

    report = {
        "schema": "phase_b_objective_selector_ablation_report_v0",
        "comparison_json": args.comparison_json,
        "raw_policy": args.raw_policy,
        "selector_policy": args.selector_policy,
        "interpretation": {
            "summary": "Objective Selector/RAM proxy v0 degraded performance relative to the raw high-level PPO policy.",
            "meaning": "This should be treated as a negative ablation, not as evidence that Objective Selector improves performance.",
            "likely_causes": [
                "naive hand-coded logit bias",
                "insufficient use of empirical robustness/uncertainty",
                "small-n episode variance",
                "no explicit speed/stability/energy trade-off score in action selection"
            ],
            "next_step": "Replace naive logit-bias selector with a robust empirical selector that scores actions using reach, stability, speed, energy proxy, and uncertainty."
        },
        "worlds": {}
    }

    for world, obj in sorted(by_world.items()):
        raw = obj.get(args.raw_policy)
        sel = obj.get(args.selector_policy)
        if raw is None or sel is None:
            continue

        delta = {
            "delta_reach_rate": float(sel["reach_rate"]) - float(raw["reach_rate"]),
            "delta_mean_min_dist": float(sel["mean_min_rel_dist"]) - float(raw["mean_min_rel_dist"]),
            "delta_mean_final_dist": float(sel["mean_final_rel_dist"]) - float(raw["mean_final_rel_dist"]),
            "delta_post_reach_drift": float(sel["mean_post_reach_drift"]) - float(raw["mean_post_reach_drift"]),
            "delta_R_v": float(sel["mean_R_v"]) - float(raw["mean_R_v"]),
            "delta_R_s": float(sel["mean_R_s"]) - float(raw["mean_R_s"]),
            "delta_R_e": float(sel.get("mean_R_e", 0.0)) - float(raw.get("mean_R_e", 0.0)),
            "delta_reward": float(sel["mean_reward"]) - float(raw["mean_reward"]),
        }

        degraded = (
            delta["delta_reach_rate"] < 0
            or delta["delta_reward"] < 0
            or delta["delta_R_v"] < 0
        )

        report["worlds"][world] = {
            "raw": raw,
            "selector": sel,
            "delta_selector_minus_raw": delta,
            "degraded": degraded,
            "recommendation": "do_not_use_proxy_v0_as_improvement" if degraded else "keep_for_further_testing"
        }

    save_json(Path(args.out_json), report)

    lines = []
    lines.append("# Phase-B Objective Selector Ablation Report")
    lines.append("")
    lines.append(f"- Raw policy: `{args.raw_policy}`")
    lines.append(f"- Selector policy: `{args.selector_policy}`")
    lines.append("")
    lines.append("This report compares the raw high-level PPO policy against the Objective Selector/RAM proxy v0.")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append("Objective Selector/RAM proxy v0 degraded performance relative to the raw PPO policy. Treat this as a negative ablation.")
    lines.append("")
    lines.append("| world | raw reach | selector reach | Δreach | raw reward | selector reward | Δreward | ΔR_v | ΔR_s | degraded |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for world, obj in report["worlds"].items():
        raw = obj["raw"]
        sel = obj["selector"]
        d = obj["delta_selector_minus_raw"]
        lines.append(
            f"| {world} | {fmt(raw['reach_rate'])} | {fmt(sel['reach_rate'])} | {fmt(d['delta_reach_rate'])} | "
            f"{fmt(raw['mean_reward'])} | {fmt(sel['mean_reward'])} | {fmt(d['delta_reward'])} | "
            f"{fmt(d['delta_R_v'])} | {fmt(d['delta_R_s'])} | {obj['degraded']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The current proxy is not a learned Objective Selector/RAM module.")
    lines.append("- It applies terrain-conditioned hand-coded logit bias, which can override useful PPO preferences.")
    lines.append("- The next selector should use empirical robustness, speed, stability, energy proxy, and uncertainty rather than a simple bias.")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
