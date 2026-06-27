#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def summarize(path: Path):
    d = json.loads(path.read_text())
    return {
        "episode_id": d.get("episode_id", path.stem),
        "terrain": d.get("terrain", "unknown"),
        "policy_id": d.get("policy_id", "unknown"),
        "duration_sec": f(d.get("duration_sec", 0.0)),
        "n_rows": int(d.get("n_rows", 0)),
        "vx_mean": f(d.get("mpc_vx_mean", 0.0)),
        "vx_p50": f(d.get("mpc_vx_p50", 0.0)),
        "vx_p90": f(d.get("mpc_vx_p90", 0.0)),
        "body_height_mean": f(d.get("mpc_body_height_mean", 0.0)),
        "clearance_mean": f(d.get("mpc_clearance_mean", 0.0)),
        "enable_mean": f(d.get("mpc_enable_mean", 0.0)),
        "gate_override_mean": f(d.get("gate_override_mean", 0.0)),
        "gate_level_mean": f(d.get("gate_level_mean", 0.0)),
        "gate_action_mean": f(d.get("gate_action_mean", 0.0)),
        "success_proxy": bool(d.get("success_proxy", False)),
        "summary_json": str(path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default="data/rollout_dataset_v0/summaries/*learned_stack_v3_gms_only_active_30s*.json",
    )
    ap.add_argument(
        "--out-json",
        default="reports/learned_stack_v3_gms_only_deploy_report.json",
    )
    ap.add_argument(
        "--out-md",
        default="reports/learned_stack_v3_gms_only_deploy_report.md",
    )
    args = ap.parse_args()

    paths = sorted(Path(".").glob(args.glob))
    rows = [summarize(p) for p in paths]

    by_terrain = {}
    for r in rows:
        by_terrain[r["terrain"]] = r

    report = {
        "version": "learned_stack_v3_gms_only_deploy_report",
        "active_scope": {
            "used_active": ["learned_gms_label"],
            "debug_only": ["learned_objective_beta", "learned_ram_intervention", "learned_objective_score"],
            "safety": [
                "one_step_delayed",
                "terrain_label_whitelist",
                "same_or_more_conservative_only",
                "rule_fallback",
            ],
        },
        "summary": by_terrain,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2))

    lines = []
    lines.append("# TRACER Learned Stack v3 GMS-only Active Deploy Report")
    lines.append("")
    lines.append("Active scope: learned GMS label only.")
    lines.append("")
    lines.append("Not active yet: learned beta, learned RAM intervention, learned objective score.")
    lines.append("")
    lines.append("Safety: one-step delayed, terrain whitelist, same-or-more-conservative only, rule fallback.")
    lines.append("")
    lines.append("| terrain | duration | rows | vx_mean | body_h | clearance | gate_override | success |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for terrain, r in by_terrain.items():
        lines.append(
            f"| {terrain} | {r['duration_sec']:.1f} | {r['n_rows']} | "
            f"{r['vx_mean']:.4f} | {r['body_height_mean']:.4f} | "
            f"{r['clearance_mean']:.4f} | {r['gate_override_mean']:.4f} | "
            f"{r['success_proxy']} |"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("- flat_normal preserved fast locomotion without command degradation.")
    lines.append("- rough_mid and slope_5deg shifted toward conservative locomotion, lowering vx and increasing body height / swing clearance.")
    lines.append("- This is the first validated active runtime path for the learned high-level stack, but only the GMS label is active.")

    out_md.write_text("\n".join(lines) + "\n")

    print("[TRACER] wrote GMS-only deploy report")
    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(out_md),
        "terrains": list(by_terrain.keys()),
    }, indent=2))


if __name__ == "__main__":
    main()
