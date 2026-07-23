#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean(vals):
    vals = [float(v) for v in vals]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--mode-active", default="j19_profile")
    ap.add_argument("--mode-baseline", default="baseline")
    ap.add_argument("--max-abs-y-margin", type=float, default=0.00)
    ap.add_argument("--mean-abs-y-margin", type=float, default=0.00)
    ap.add_argument("--abs-delta-y-margin", type=float, default=0.00)
    ap.add_argument("--require-all-metrics-nonworse", action="store_true", default=True)
    args = ap.parse_args()

    rows = read_csv(args.segment_csv)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["mode"], r["context"])].append(r)

    contexts = sorted(set(r["context"] for r in rows))

    context_rules = {}
    delta_rows = []

    for ctx in contexts:
        b = groups.get((args.mode_baseline, ctx), [])
        a = groups.get((args.mode_active, ctx), [])

        if not b or not a:
            context_rules[ctx] = {
                "decision": "block",
                "reason": "missing_baseline_or_active_data",
                "n_baseline": len(b),
                "n_active": len(a),
            }
            continue

        b_max = mean([r["max_abs_y"] for r in b])
        a_max = mean([r["max_abs_y"] for r in a])
        b_mean = mean([r["mean_abs_y"] for r in b])
        a_mean = mean([r["mean_abs_y"] for r in a])
        b_dy = mean([r["abs_delta_y"] for r in b])
        a_dy = mean([r["abs_delta_y"] for r in a])

        d_max = a_max - b_max
        d_mean = a_mean - b_mean
        d_dy = a_dy - b_dy

        nonworse_max = d_max <= args.max_abs_y_margin
        nonworse_mean = d_mean <= args.mean_abs_y_margin
        nonworse_dy = d_dy <= args.abs_delta_y_margin

        allow = nonworse_max and nonworse_mean and nonworse_dy

        # Additional hard protection for contexts where current J19 did not actually project active commands.
        active_projected_mean = mean([r["j7_projected"] for r in a])
        active_accepted_mean = mean([r["j7_accepted"] for r in a])
        has_active_projection = active_projected_mean > 1.0

        if allow and has_active_projection:
            decision = "allow"
            reason = "active_nonworse_and_projected"
        elif allow and not has_active_projection:
            decision = "protected_empirical"
            reason = "nonworse_but_no_active_projection_in_this_context"
        else:
            decision = "block"
            reason = "active_degrades_segment_metrics"

        context_rules[ctx] = {
            "decision": decision,
            "reason": reason,
            "n_baseline": len(b),
            "n_active": len(a),
            "baseline": {
                "max_abs_y_mean": b_max,
                "mean_abs_y_mean": b_mean,
                "abs_delta_y_mean": b_dy,
            },
            "active": {
                "max_abs_y_mean": a_max,
                "mean_abs_y_mean": a_mean,
                "abs_delta_y_mean": a_dy,
                "j7_accepted_mean": active_accepted_mean,
                "j7_projected_mean": active_projected_mean,
            },
            "delta_active_minus_baseline": {
                "max_abs_y_mean": d_max,
                "mean_abs_y_mean": d_mean,
                "abs_delta_y_mean": d_dy,
            },
            "checks": {
                "nonworse_max_abs_y": nonworse_max,
                "nonworse_mean_abs_y": nonworse_mean,
                "nonworse_abs_delta_y": nonworse_dy,
                "has_active_projection": has_active_projection,
            },
        }

        delta_rows.append((ctx, d_max, d_mean, d_dy, active_accepted_mean, active_projected_mean, decision, reason))

    # Conservative global decision:
    # Do not promote a deployable active profile unless the actually-active context is allowed.
    actually_active_contexts = [
        ctx for ctx, rule in context_rules.items()
        if rule.get("active", {}).get("j7_projected_mean", 0.0) > 1.0
    ]
    allowed_active_contexts = [
        ctx for ctx in actually_active_contexts
        if context_rules[ctx]["decision"] == "allow"
    ]

    deploy_performance_policy = bool(allowed_active_contexts) and set(allowed_active_contexts) == set(actually_active_contexts)

    risk_rule = {
        "phase": "K3",
        "name": "conservative_risk_rule_from_k2b_v0",
        "source_segment_csv": args.segment_csv,
        "baseline_mode": args.mode_baseline,
        "active_mode": args.mode_active,
        "margins": {
            "max_abs_y_margin": args.max_abs_y_margin,
            "mean_abs_y_margin": args.mean_abs_y_margin,
            "abs_delta_y_margin": args.abs_delta_y_margin,
        },
        "global_decision": {
            "deploy_performance_policy": deploy_performance_policy,
            "actually_active_contexts": actually_active_contexts,
            "allowed_active_contexts": allowed_active_contexts,
            "blocked_active_contexts": [
                ctx for ctx in actually_active_contexts
                if ctx not in allowed_active_contexts
            ],
            "recommended_runtime_mode": (
                "active_allowed" if deploy_performance_policy else "empirical_default_active_shadow_only"
            ),
        },
        "context_rules": context_rules,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(risk_rule, indent=2, sort_keys=True) + "\n")

    lines = []
    lines.append("# TRACER Phase-K3 Conservative Risk Rule Summary v0")
    lines.append("")
    lines.append(f"- source segment csv: `{args.segment_csv}`")
    lines.append(f"- output rule json: `{args.out_json}`")
    lines.append("")
    lines.append("## Global decision")
    lines.append("")
    lines.append(f"- deploy performance policy: `{deploy_performance_policy}`")
    lines.append(f"- recommended runtime mode: `{risk_rule['global_decision']['recommended_runtime_mode']}`")
    lines.append(f"- actually active contexts: `{actually_active_contexts}`")
    lines.append(f"- allowed active contexts: `{allowed_active_contexts}`")
    lines.append(f"- blocked active contexts: `{risk_rule['global_decision']['blocked_active_contexts']}`")
    lines.append("")
    lines.append("## Context-level decision")
    lines.append("")
    lines.append("| context | Δ max_abs_y | Δ mean_abs_y | Δ abs_delta_y | j7 accepted mean | j7 projected mean | decision | reason |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")

    for ctx, d_max, d_mean, d_dy, acc, proj, decision, reason in sorted(delta_rows):
        lines.append(
            f"| {ctx} | {d_max:+.6f} | {d_mean:+.6f} | {d_dy:+.6f} | "
            f"{acc:.2f} | {proj:.2f} | {decision} | {reason} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- K3 uses a conservative non-worsening rule.")
    lines.append("- A context is allowed only if active metrics are non-worse than baseline and active projection actually occurred there.")
    lines.append("- If the actually active context is degraded, the profile is not promoted as a deployable performance policy.")
    lines.append("- The J19 profile can still be kept as a scaffold/debug active-routing profile.")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")

    print(f"[TRACER:K3] wrote {args.out_json}")
    print(f"[TRACER:K3] wrote {args.out_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
