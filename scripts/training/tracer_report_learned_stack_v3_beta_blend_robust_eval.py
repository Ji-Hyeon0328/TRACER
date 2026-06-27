#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean, pstdev


TERRAINS = ["flat_normal", "rough_mid", "slope_5deg"]


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def stat(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(xs),
        "std": pstdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def fmt_stat(s, nd=4):
    if s["mean"] is None:
        return "n/a"
    return f"{s['mean']:.{nd}f}±{s['std']:.{nd}f}"


def find_summaries(terrain: str, limit: int):
    pattern = (
        "data/rollout_dataset_v0/summaries/"
        f"{terrain}_theta_mlp_udp_v0_learned_stack_v3_beta_blend_robust30_{terrain}_r*.json"
    )
    paths = [Path(p) for p in glob.glob(pattern)]
    paths = sorted(paths, key=lambda p: p.stat().st_mtime)
    if limit > 0:
        paths = paths[-limit:]
    return paths


def summarize_terrain(terrain: str, limit: int):
    paths = find_summaries(terrain, limit)
    rows = []
    for p in paths:
        d = load_json(p)
        if d is None:
            continue
        d["_summary_path"] = str(p)
        rows.append(d)

    success = [1.0 if r.get("success_proxy") else 0.0 for r in rows]

    out = {
        "terrain": terrain,
        "n": len(rows),
        "success_rate": mean(success) if success else 0.0,
        "paths": [r["_summary_path"] for r in rows],
        "stats": {
            "mpc_vx_mean": stat([r.get("mpc_vx_mean") for r in rows]),
            "mpc_body_height_mean": stat([r.get("mpc_body_height_mean") for r in rows]),
            "mpc_clearance_mean": stat([r.get("mpc_clearance_mean") for r in rows]),
            "gate_override_mean": stat([r.get("gate_override_mean") for r in rows]),
            "ram_recovery_needed_mean": stat([r.get("ram_recovery_needed_mean") for r in rows]),
            "ram_run_fallen_mean": stat([r.get("ram_run_fallen_mean") for r in rows]),
        },
        "episodes": [
            {
                "episode_id": r.get("episode_id"),
                "success_proxy": r.get("success_proxy"),
                "mpc_vx_mean": r.get("mpc_vx_mean"),
                "mpc_body_height_mean": r.get("mpc_body_height_mean"),
                "mpc_clearance_mean": r.get("mpc_clearance_mean"),
                "gate_override_mean": r.get("gate_override_mean"),
                "summary_path": r.get("_summary_path"),
            }
            for r in rows
        ],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument(
        "--out-json",
        default="reports/learned_stack_v3_beta_blend_robust_eval_report.json",
    )
    ap.add_argument(
        "--out-md",
        default="reports/learned_stack_v3_beta_blend_robust_eval_report.md",
    )
    args = ap.parse_args()

    terrains = {t: summarize_terrain(t, args.limit) for t in TERRAINS}
    overall_pass = all(v["n"] >= args.limit and v["success_rate"] >= 1.0 for v in terrains.values())

    report = {
        "version": "learned_stack_v3_beta_blend_robust_eval_v0",
        "active_scope": {
            "active": [
                "learned_gms_label",
                "learned_objective_beta_low_alpha_blend",
            ],
            "safety": [
                "alpha_limited",
                "terrain_specific_delta_limit",
                "gms_only_safety_path",
                "ram_recovery_not_active",
            ],
            "not_active": [
                "full_beta_replacement",
                "learned_ram_hard_gate",
                "learned_objective_score_command_selector",
            ],
        },
        "overall_pass": overall_pass,
        "terrains": terrains,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2))

    lines = []
    lines.append("# TRACER Learned Stack v3 Beta Blend Robust Eval Report")
    lines.append("")
    lines.append("Active scope: learned GMS label + learned objective beta low-alpha blend.")
    lines.append("")
    lines.append("Safety: alpha-limited, terrain-specific delta-limited, GMS-only safety path; RAM recovery is not active.")
    lines.append("")
    lines.append(f"Overall status: {'PASS' if overall_pass else 'CHECK'}")
    lines.append("")
    lines.append("| terrain | n | success | vx_mean | body_h | clearance | gate_override | ram_recovery |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for terrain in TERRAINS:
        s = terrains[terrain]
        st = s["stats"]
        lines.append(
            f"| {terrain} | "
            f"{s['n']} | "
            f"{s['success_rate']:.3f} | "
            f"{fmt_stat(st['mpc_vx_mean'])} | "
            f"{fmt_stat(st['mpc_body_height_mean'])} | "
            f"{fmt_stat(st['mpc_clearance_mean'])} | "
            f"{fmt_stat(st['gate_override_mean'])} | "
            f"{fmt_stat(st['ram_recovery_needed_mean'])} |"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("- This report validates low-alpha beta blend stability over repeated rollouts.")
    lines.append("- It does not validate full learned beta replacement.")
    lines.append("- It does not activate learned RAM as a recovery gate.")

    out_md.write_text("\n".join(lines) + "\n")

    print("[TRACER] wrote beta blend robust eval report")
    print(json.dumps({
        "overall_pass": overall_pass,
        "out_json": str(out_json),
        "out_md": str(out_md),
    }, indent=2))


if __name__ == "__main__":
    main()
