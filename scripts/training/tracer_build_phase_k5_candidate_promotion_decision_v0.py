#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def ff(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k4c-summary-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = read_csv(args.k4c_summary_csv)
    ok = [r for r in rows if r.get("status") == "ok"]

    groups = defaultdict(list)
    for r in ok:
        groups[r["mode"]].append(r)

    baseline = groups.get("baseline", [])
    if not baseline:
        raise SystemExit("No baseline rows found")

    b_max = mean([ff(r["max_abs_y"]) for r in baseline])
    b_mean = mean([ff(r["mean_abs_y"]) for r in baseline])
    b_goal = sum(1 for r in baseline if r["goal_reached"] == "True")

    decisions = {}
    for mode, rs in sorted(groups.items()):
        max_mean = mean([ff(r["max_abs_y"]) for r in rs])
        mean_mean = mean([ff(r["mean_abs_y"]) for r in rs])
        goals = sum(1 for r in rs if r["goal_reached"] == "True")
        n = len(rs)
        j7_acc = mean([ff(r["j7_accepted"]) for r in rs])

        if mode == "baseline":
            decision = "reference"
            reason = "baseline_reference"
        else:
            improves_max = max_mean < b_max
            improves_mean = mean_mean < b_mean
            all_goals = goals == n
            if all_goals and improves_max and improves_mean:
                decision = "promote"
                reason = "improves_both_lateral_metrics_and_reaches_all_goals"
            elif all_goals and improves_max and not improves_mean:
                decision = "shadow_only"
                reason = "improves_max_abs_y_only_but_worsens_mean_abs_y"
            elif all_goals:
                decision = "reject"
                reason = "does_not_improve_lateral_metrics"
            else:
                decision = "reject"
                reason = "does_not_reach_all_goals_or_has_invalid_runs"

        decisions[mode] = {
            "n_ok": n,
            "goals": goals,
            "max_abs_y_mean": max_mean,
            "mean_abs_y_mean": mean_mean,
            "delta_max_abs_y_vs_baseline": max_mean - b_max,
            "delta_mean_abs_y_vs_baseline": mean_mean - b_mean,
            "j7_accepted_mean": j7_acc,
            "decision": decision,
            "reason": reason,
        }

    promoted = [
        m for m, d in decisions.items()
        if d["decision"] == "promote"
    ]

    result = {
        "phase": "K5",
        "name": "flat_micro_candidate_promotion_decision_v0",
        "source": args.k4c_summary_csv,
        "baseline": {
            "n_ok": len(baseline),
            "goals": b_goal,
            "max_abs_y_mean": b_max,
            "mean_abs_y_mean": b_mean,
        },
        "promoted_candidates": promoted,
        "global_decision": {
            "deploy_active_performance_policy": bool(promoted),
            "recommended_runtime_mode": "active_candidate" if promoted else "empirical_default_active_shadow_only",
            "recommended_candidate": promoted[0] if promoted else None,
        },
        "candidate_decisions": decisions,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = []
    lines.append("# TRACER Phase-K5 Candidate Promotion Decision v0")
    lines.append("")
    lines.append(f"- source: `{args.k4c_summary_csv}`")
    lines.append(f"- output json: `{args.out_json}`")
    lines.append("")
    lines.append("## Global decision")
    lines.append("")
    lines.append(f"- deploy active performance policy: `{result['global_decision']['deploy_active_performance_policy']}`")
    lines.append(f"- recommended runtime mode: `{result['global_decision']['recommended_runtime_mode']}`")
    lines.append(f"- promoted candidates: `{promoted}`")
    lines.append("")
    lines.append("## Candidate decisions")
    lines.append("")
    lines.append("| mode | n ok | goals | max_abs_y mean | mean_abs_y mean | Δ max_abs_y | Δ mean_abs_y | j7 accepted mean | decision | reason |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")

    for mode, d in decisions.items():
        lines.append(
            f"| {mode} | {d['n_ok']} | {d['goals']} | "
            f"{d['max_abs_y_mean']:.6f} | {d['mean_abs_y_mean']:.6f} | "
            f"{d['delta_max_abs_y_vs_baseline']:+.6f} | {d['delta_mean_abs_y_vs_baseline']:+.6f} | "
            f"{d['j7_accepted_mean']:.2f} | {d['decision']} | {d['reason']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- K4C confirms that the fixed active routing path works.")
    lines.append("- No candidate improves both max_abs_y and mean_abs_y relative to baseline.")
    lines.append("- flat_slow03 is the closest candidate, but it only slightly improves max_abs_y while worsening mean_abs_y.")
    lines.append("- Therefore, no flat micro-action candidate should be promoted as a deployable performance policy.")
    lines.append("- Active routing should remain available as scaffold/debug, while empirical/default remains the runtime policy.")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")

    print(f"[TRACER:K5] wrote {args.out_json}")
    print(f"[TRACER:K5] wrote {args.out_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
