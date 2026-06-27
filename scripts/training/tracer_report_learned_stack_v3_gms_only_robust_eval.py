#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "mpc_vx_mean",
    "mpc_body_height_mean",
    "mpc_clearance_mean",
    "mpc_enable_mean",
    "gate_override_mean",
    "gate_level_mean",
    "gate_action_mean",
]


def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def mean(xs):
    return sum(xs) / max(1, len(xs))


def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def summarize_group(rows):
    out = {
        "n": len(rows),
        "success_count": sum(1 for r in rows if bool(r.get("success_proxy", False))),
        "success_rate": sum(1 for r in rows if bool(r.get("success_proxy", False))) / max(1, len(rows)),
        "episodes": [r.get("episode_id", "unknown") for r in rows],
    }

    for key in FIELDS:
        vals = [f(r.get(key, 0.0)) for r in rows]
        out[key] = {
            "mean": mean(vals),
            "std": std(vals),
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
        }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default="data/rollout_dataset_v0/summaries/*learned_stack_v3_gms_only_robust30*.json",
    )
    ap.add_argument(
        "--out-json",
        default="reports/learned_stack_v3_gms_only_robust_eval_report.json",
    )
    ap.add_argument(
        "--out-md",
        default="reports/learned_stack_v3_gms_only_robust_eval_report.md",
    )
    args = ap.parse_args()

    paths = sorted(Path(".").glob(args.glob))
    rows = []
    for p in paths:
        try:
            d = json.loads(p.read_text())
            d["_summary_json"] = str(p)
            rows.append(d)
        except Exception as e:
            print(f"[TRACER] skip {p}: {e!r}")

    groups = defaultdict(list)
    for r in rows:
        groups[str(r.get("terrain", "unknown"))].append(r)

    summary = {terrain: summarize_group(rs) for terrain, rs in sorted(groups.items())}

    report = {
        "version": "learned_stack_v3_gms_only_robust_eval_report",
        "glob": args.glob,
        "active_scope": {
            "used_active": ["learned_gms_label"],
            "debug_only": [
                "learned_objective_beta",
                "learned_ram_intervention",
                "learned_objective_score",
            ],
            "safety": [
                "one_step_delayed",
                "terrain_label_whitelist",
                "same_or_more_conservative_only",
                "rule_fallback",
            ],
        },
        "n_total": len(rows),
        "pass": all(v.get("success_rate", 0.0) >= 1.0 for v in summary.values()),
        "summary": summary,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2))

    lines = []
    lines.append("# TRACER Learned Stack v3 GMS-only Robust Eval Report")
    lines.append("")
    lines.append("Active scope: learned GMS label only.")
    lines.append("")
    lines.append("Not active yet: learned beta, learned RAM intervention, learned objective score.")
    lines.append("")
    lines.append("Safety: one-step delayed, terrain whitelist, same-or-more-conservative only, rule fallback.")
    lines.append("")
    lines.append("| terrain | n | success | vx_mean | body_h | clearance | gate_override |")
    passed = all(v.get("success_rate", 0.0) >= 1.0 for v in summary.values())
    lines.append("")
    lines.append(f"Overall status: {'PASS' if passed else 'FAIL'}")
    lines.append("")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for terrain, s in summary.items():
        lines.append(
            f"| {terrain} | {s['n']} | {s['success_rate']:.3f} | "
            f"{s['mpc_vx_mean']['mean']:.4f}±{s['mpc_vx_mean']['std']:.4f} | "
            f"{s['mpc_body_height_mean']['mean']:.4f}±{s['mpc_body_height_mean']['std']:.4f} | "
            f"{s['mpc_clearance_mean']['mean']:.4f}±{s['mpc_clearance_mean']['std']:.4f} | "
            f"{s['gate_override_mean']['mean']:.4f}±{s['gate_override_mean']['std']:.4f} |"
        )

    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("- flat_normal should preserve fast locomotion: vx near 0.28, low gate override, success rate 1.0.")
    lines.append("- rough_mid and slope_5deg should shift toward conservative locomotion: lower vx, higher body height and clearance.")
    lines.append("- This report evaluates the active GMS-only path; if any terrain has success rate below 1.0, treat it as a failure-finding report, not a validation report.")

    out_md.write_text("\n".join(lines) + "\n")

    print("[TRACER] wrote learned stack v3 GMS-only robust eval report")
    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(out_md),
        "n_total": len(rows),
        "pass": all(v.get("success_rate", 0.0) >= 1.0 for v in summary.values()),
        "terrains": list(summary.keys()),
    }, indent=2))


if __name__ == "__main__":
    main()
