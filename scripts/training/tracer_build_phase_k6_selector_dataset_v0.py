#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict


def ff(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(Path(path).read_text())


def mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def mode_family(mode):
    if mode == "baseline":
        return "empirical_baseline"
    if mode == "j19_profile":
        return "j19_flat_noop_profile"
    if mode.startswith("flat_"):
        return "flat_micro_action"
    return "unknown"


def candidate_theta_desc(mode):
    table = {
        "baseline": "empirical_D7_D6_gated",
        "j19_profile": "flat_noop_active_profile",
        "flat_noop": "flat theta no-op",
        "flat_slow03": "flat vx_scale 0.97",
        "flat_clear03": "flat clearance +0.003",
        "flat_slow03_clear03": "flat vx_scale 0.97 + clearance +0.003",
    }
    return table.get(mode, "unknown")


def row_to_common(source_phase, r, decision="unknown", reason="unknown"):
    status = r.get("status", "")
    mode = r.get("mode", "")
    goal = r.get("goal_reached", "False") == "True"

    max_abs_y = ff(r.get("max_abs_y"))
    mean_abs_y = ff(r.get("mean_abs_y"))
    final_x = ff(r.get("final_x"))
    final_y = ff(r.get("final_y"))
    j7_accepted = ff(r.get("j7_accepted")) or 0.0
    j7_projected = ff(r.get("j7_projected")) or 0.0
    j7_failsafe = ff(r.get("j7_failsafe")) or 0.0
    flat_rows = ff(r.get("flat_rows")) or 0.0
    flat_accepted = ff(r.get("flat_accepted")) or 0.0

    active_flat_accept_rate = flat_accepted / flat_rows if flat_rows > 0 else 0.0
    active_used = j7_projected > 1.0 or j7_accepted > 1.0

    # Lower is better. This is a conservative ranking score, not a learned reward.
    invalid_penalty = 1.0 if status != "ok" else 0.0
    goal_penalty = 1.0 if not goal else 0.0
    nan_penalty = 1.0 if max_abs_y is None or mean_abs_y is None else 0.0
    risk_score = (
        (max_abs_y if max_abs_y is not None else 1.0)
        + 0.5 * (mean_abs_y if mean_abs_y is not None else 1.0)
        + 2.0 * goal_penalty
        + 2.0 * invalid_penalty
        + 1.0 * nan_penalty
    )

    return {
        "source_phase": source_phase,
        "mode": mode,
        "idx": r.get("idx", ""),
        "mode_family": mode_family(mode),
        "candidate_theta_desc": candidate_theta_desc(mode),
        "status": status,
        "goal_reached": int(goal),
        "final_x": final_x,
        "final_y": final_y,
        "max_abs_y": max_abs_y,
        "mean_abs_y": mean_abs_y,
        "risk_score_lower_better": risk_score,
        "active_used": int(active_used),
        "j7_accepted": j7_accepted,
        "j7_projected": j7_projected,
        "j7_failsafe": j7_failsafe,
        "flat_rows": flat_rows,
        "flat_accepted": flat_accepted,
        "flat_accept_rate": active_flat_accept_rate,
        "decision": decision,
        "decision_reason": reason,
        "manifest": r.get("manifest", ""),
        "d5_log_dir": r.get("d5_log_dir", ""),
        "j7_csv": r.get("j7_csv", ""),
        "run_dir": r.get("run_dir", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k1-summary-csv", required=True)
    ap.add_argument("--k4c-summary-csv", required=True)
    ap.add_argument("--k3-risk-json", required=True)
    ap.add_argument("--k5-decision-json", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    k1_rows = read_csv(args.k1_summary_csv)
    k4c_rows = read_csv(args.k4c_summary_csv)
    k3 = read_json(args.k3_risk_json)
    k5 = read_json(args.k5_decision_json)

    out_rows = []

    for r in k1_rows:
        mode = r.get("mode", "")
        if mode == "baseline":
            decision = "reference"
            reason = "baseline_reference"
        elif mode == "j19_profile":
            decision = "reject"
            reason = k3.get("global_decision", {}).get("recommended_runtime_mode", "k3_rejected")
        else:
            decision = "unknown"
            reason = "unknown_k1_mode"
        out_rows.append(row_to_common("K1", r, decision, reason))

    k5_decisions = k5.get("candidate_decisions", {})
    for r in k4c_rows:
        mode = r.get("mode", "")
        d = k5_decisions.get(mode, {})
        decision = d.get("decision", "unknown")
        reason = d.get("reason", "unknown")
        out_rows.append(row_to_common("K4C", r, decision, reason))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "source_phase", "mode", "idx", "mode_family", "candidate_theta_desc",
        "status", "goal_reached", "final_x", "final_y",
        "max_abs_y", "mean_abs_y", "risk_score_lower_better",
        "active_used", "j7_accepted", "j7_projected", "j7_failsafe",
        "flat_rows", "flat_accepted", "flat_accept_rate",
        "decision", "decision_reason",
        "manifest", "d5_log_dir", "j7_csv", "run_dir",
    ]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    groups = defaultdict(list)
    for r in out_rows:
        groups[(r["source_phase"], r["mode"])].append(r)

    lines = []
    lines.append("# TRACER Phase-K6 Selector Dataset Summary v0")
    lines.append("")
    lines.append(f"- K1 source: `{args.k1_summary_csv}`")
    lines.append(f"- K4C source: `{args.k4c_summary_csv}`")
    lines.append(f"- K3 risk rule: `{args.k3_risk_json}`")
    lines.append(f"- K5 decision: `{args.k5_decision_json}`")
    lines.append(f"- output dataset: `{args.out_csv}`")
    lines.append("")
    lines.append("## Group summary")
    lines.append("")
    lines.append("| source | mode | n | ok | goals | active used mean | max_abs_y mean | mean_abs_y mean | risk score mean | decision |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")

    for key in sorted(groups.keys()):
        rs = groups[key]
        source, mode = key
        ok = sum(1 for r in rs if r["status"] == "ok")
        goals = sum(int(r["goal_reached"]) for r in rs)
        active_mean = mean([float(r["active_used"]) for r in rs])
        max_mean = mean([r["max_abs_y"] for r in rs if r["max_abs_y"] is not None])
        mean_y = mean([r["mean_abs_y"] for r in rs if r["mean_abs_y"] is not None])
        risk = mean([r["risk_score_lower_better"] for r in rs])
        decision = sorted(set(r["decision"] for r in rs))
        lines.append(
            f"| {source} | {mode} | {len(rs)} | {ok} | {goals} | "
            f"{active_mean:.3f} | {max_mean:.6f} | {mean_y:.6f} | {risk:.6f} | `{decision}` |"
        )

    lines.append("")
    lines.append("## Dataset usage")
    lines.append("")
    lines.append("- This dataset is not yet enough to train a strong RL policy.")
    lines.append("- It is useful for a conservative selector/risk classifier prototype.")
    lines.append("- Positive deploy labels are intentionally absent because K5 promoted no active candidate.")
    lines.append("- `flat_slow03` can be kept as a shadow-only near-miss sample.")
    lines.append("- The next step is K7: build a conservative selector that predicts `empirical`, `shadow_only`, or `reject` from context/action/outcome features.")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")

    print(f"[TRACER:K6] wrote {args.out_csv}")
    print(f"[TRACER:K6] wrote {args.out_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
