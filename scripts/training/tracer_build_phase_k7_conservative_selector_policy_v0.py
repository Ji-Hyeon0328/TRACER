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
    ap.add_argument("--k6-dataset-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = read_csv(args.k6_dataset_csv)

    groups = defaultdict(list)
    for r in rows:
        groups[r["mode"]].append(r)

    mode_rules = {}

    for mode, rs in sorted(groups.items()):
        decisions = sorted(set(r["decision"] for r in rs))
        n = len(rs)
        ok = sum(1 for r in rs if r["status"] == "ok")
        goals = sum(int(r["goal_reached"]) for r in rs)
        active_used_mean = mean([ff(r["active_used"]) for r in rs])
        max_abs_y_mean = mean([ff(r["max_abs_y"]) for r in rs])
        mean_abs_y_mean = mean([ff(r["mean_abs_y"]) for r in rs])
        risk_score_mean = mean([ff(r["risk_score_lower_better"]) for r in rs])
        j7_acc_mean = mean([ff(r["j7_accepted"]) for r in rs])

        if mode == "baseline":
            selector_action = "empirical"
            deploy_allowed = True
            reason = "baseline_reference"
        elif "promote" in decisions:
            selector_action = "active"
            deploy_allowed = True
            reason = "candidate_promoted_by_prior_decision"
        elif "shadow_only" in decisions:
            selector_action = "shadow_only"
            deploy_allowed = False
            reason = "near_miss_but_not_safe_to_deploy"
        else:
            selector_action = "reject"
            deploy_allowed = False
            reason = "rejected_by_prior_decision"

        mode_rules[mode] = {
            "selector_action": selector_action,
            "deploy_allowed": deploy_allowed,
            "reason": reason,
            "n": n,
            "ok": ok,
            "goals": goals,
            "active_used_mean": active_used_mean,
            "max_abs_y_mean": max_abs_y_mean,
            "mean_abs_y_mean": mean_abs_y_mean,
            "risk_score_mean": risk_score_mean,
            "j7_accepted_mean": j7_acc_mean,
            "input_decisions": decisions,
        }

    active_modes = [
        m for m, r in mode_rules.items()
        if r["selector_action"] == "active"
    ]
    shadow_modes = [
        m for m, r in mode_rules.items()
        if r["selector_action"] == "shadow_only"
    ]
    rejected_modes = [
        m for m, r in mode_rules.items()
        if r["selector_action"] == "reject"
    ]

    policy = {
        "phase": "K7",
        "name": "conservative_selector_policy_v0",
        "source_dataset": args.k6_dataset_csv,
        "global_decision": {
            "default_runtime_action": "empirical",
            "deploy_active_performance_policy": bool(active_modes),
            "active_modes": active_modes,
            "shadow_only_modes": shadow_modes,
            "rejected_modes": rejected_modes,
            "runtime_note": (
                "No active candidate is deployable yet. "
                "Use empirical/default control for runtime; keep near-miss active candidates only in shadow."
            ),
        },
        "mode_rules": mode_rules,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")

    lines = []
    lines.append("# TRACER Phase-K7 Conservative Selector Policy v0")
    lines.append("")
    lines.append(f"- source dataset: `{args.k6_dataset_csv}`")
    lines.append(f"- output policy: `{args.out_json}`")
    lines.append("")
    lines.append("## Global decision")
    lines.append("")
    lines.append(f"- default runtime action: `{policy['global_decision']['default_runtime_action']}`")
    lines.append(f"- deploy active performance policy: `{policy['global_decision']['deploy_active_performance_policy']}`")
    lines.append(f"- active modes: `{active_modes}`")
    lines.append(f"- shadow-only modes: `{shadow_modes}`")
    lines.append(f"- rejected modes: `{rejected_modes}`")
    lines.append("")
    lines.append("## Mode rules")
    lines.append("")
    lines.append("| mode | selector action | deploy allowed | n | ok | goals | max_abs_y mean | mean_abs_y mean | risk score mean | j7 accepted mean | reason |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for mode, r in mode_rules.items():
        lines.append(
            f"| {mode} | {r['selector_action']} | {r['deploy_allowed']} | "
            f"{r['n']} | {r['ok']} | {r['goals']} | "
            f"{r['max_abs_y_mean']:.6f} | {r['mean_abs_y_mean']:.6f} | "
            f"{r['risk_score_mean']:.6f} | {r['j7_accepted_mean']:.2f} | "
            f"{r['reason']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- K7 does not promote any active performance policy.")
    lines.append("- The runtime-safe choice remains empirical/default control.")
    lines.append("- `flat_slow03` is retained only as a shadow-only near-miss candidate.")
    lines.append("- This policy is a conservative blocker/selector scaffold, not a learned RL meta-planner.")
    lines.append("- Next step should be either Phase-K closure or a broader learned selector dataset with more diverse terrain/action candidates.")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")

    print(f"[TRACER:K7] wrote {args.out_json}")
    print(f"[TRACER:K7] wrote {args.out_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
