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
    ap.add_argument("--comparison-json", default="reports/phase_b_reach_terminated_policy_comparison_v0.json")
    ap.add_argument("--tracer-policy", default="tracer_proxy_ppo_argmax_v1")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    comp = load_json(args.comparison_json)
    rows = comp["rows"]

    by_world = {}
    for r in rows:
        by_world.setdefault(r["world"], []).append(r)

    report = {
        "schema": "phase_b_policy_oracle_gap_report_v0",
        "comparison_json": args.comparison_json,
        "tracer_policy": args.tracer_policy,
        "interpretation": {
            "purpose": "Compare the current TRACER high-level policy against the best fixed-action baseline per terrain.",
            "main_metric": "mean_reward from the TRACER slide-style proxy reward, with reach_rate and R_v/R_s reported separately.",
            "note": "The oracle fixed baseline is not a deployable policy; it is an analysis tool showing which terrain-action preference the high-level planner should learn."
        },
        "worlds": {}
    }

    for world, wrs in sorted(by_world.items()):
        tracer = next((r for r in wrs if r["policy"] == args.tracer_policy), None)
        fixed = [r for r in wrs if r["policy"].startswith("fixed_")]

        if not fixed:
            continue

        best_fixed = sorted(
            fixed,
            key=lambda r: (
                float(r.get("mean_reward", -1e9)),
                float(r.get("reach_rate", -1e9)),
                -float(r.get("mean_min_rel_dist", 1e9)),
            ),
            reverse=True,
        )[0]

        if tracer is None:
            gap = None
            tracer_beats_oracle = None
        else:
            gap = float(tracer["mean_reward"]) - float(best_fixed["mean_reward"])
            tracer_beats_oracle = gap >= 0.0

        recommendation = "keep_current"
        if tracer is not None and not tracer_beats_oracle:
            recommendation = f"shift_preference_toward_{best_fixed['policy'].replace('fixed_', '')}"

        # Special note for sponge: high reach + poor R_s should be treated as low-level hold limitation.
        limitation = ""
        if world == "tracer_sponge_firm_flat" and tracer is not None:
            if float(tracer.get("reach_rate", 0.0)) >= 1.0 and float(tracer.get("mean_R_s", 0.0)) < 0.0:
                limitation = "TRACER reaches the goal neighborhood, but R_s is negative due to post-reach drift. Treat this as frozen low-level hold limitation."

        report["worlds"][world] = {
            "tracer": tracer,
            "best_fixed": best_fixed,
            "tracer_minus_best_fixed_reward": gap,
            "tracer_beats_oracle_fixed": tracer_beats_oracle,
            "recommendation": recommendation,
            "limitation": limitation,
        }

    save_json(Path(args.out_json), report)

    lines = []
    lines.append("# Phase-B Policy Oracle Gap Report")
    lines.append("")
    lines.append(f"- TRACER policy: `{args.tracer_policy}`")
    lines.append(f"- Source: `{args.comparison_json}`")
    lines.append("")
    lines.append("This report compares the current TRACER high-level policy against the best fixed-action baseline per terrain.")
    lines.append("The fixed-action oracle is an analysis tool, not a deployable adaptive policy.")
    lines.append("")
    lines.append("| world | TRACER reward | TRACER reach | best fixed policy | best fixed reward | best fixed reach | gap | recommendation |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---|")

    for world, obj in report["worlds"].items():
        t = obj["tracer"]
        b = obj["best_fixed"]
        lines.append(
            f"| {world} | {fmt(t.get('mean_reward') if t else None)} | "
            f"{fmt(t.get('reach_rate') if t else None)} | "
            f"{b.get('policy')} | {fmt(b.get('mean_reward'))} | {fmt(b.get('reach_rate'))} | "
            f"{fmt(obj.get('tracer_minus_best_fixed_reward'))} | {obj.get('recommendation')} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for world, obj in report["worlds"].items():
        if obj.get("limitation"):
            lines.append(f"- `{world}`: {obj['limitation']}")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
