#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


KEYS = ["motion", "stability", "energy"]


def load_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def stat(xs):
    if not xs:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(xs),
        "std": pstdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def summarize_terrain(terrain: str, path: Path):
    rows = load_rows(path)
    vals = []

    reasons = {}
    for r in rows:
        bs = r.get("learned_stack_v3_beta_shadow") or {}
        reason = str(bs.get("reason"))
        reasons[reason] = reasons.get(reason, 0) + 1
        if reason == "shadow_only":
            vals.append(bs)

    out = {
        "terrain": terrain,
        "path": str(path),
        "rows": len(rows),
        "shadow_rows": len(vals),
        "reasons": reasons,
        "rule_beta": {},
        "learned_raw": {},
        "learned_clamped": {},
        "delta": {},
    }

    for field in ["rule_beta", "learned_beta_raw", "learned_beta_clamped", "delta"]:
        target_name = {
            "rule_beta": "rule_beta",
            "learned_beta_raw": "learned_raw",
            "learned_beta_clamped": "learned_clamped",
            "delta": "delta",
        }[field]

        for k in KEYS:
            xs = []
            for v in vals:
                d = v.get(field) or {}
                if k in d:
                    xs.append(float(d[k]))
            out[target_name][k] = stat(xs)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prefix",
        default="/tmp/tracer_highlevel_debug_beta_shadow_rawfix",
    )
    ap.add_argument(
        "--out-json",
        default="reports/learned_stack_v3_beta_shadow_report.json",
    )
    ap.add_argument(
        "--out-md",
        default="reports/learned_stack_v3_beta_shadow_report.md",
    )
    args = ap.parse_args()

    terrains = ["flat_normal", "rough_mid", "slope_5deg"]
    summary = {}

    for terrain in terrains:
        path = Path(f"{args.prefix}_{terrain}.txt")
        summary[terrain] = summarize_terrain(terrain, path)

    report = {
        "version": "learned_stack_v3_beta_shadow_report",
        "active_scope": {
            "active": ["learned_gms_label"],
            "shadow_only": ["learned_objective_beta"],
            "not_active": ["beta_command_influence", "ram_triggered_recovery"],
        },
        "summary": summary,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))

    lines = []
    lines.append("# TRACER Learned Stack v3 Beta Shadow Report")
    lines.append("")
    lines.append("Active component: learned GMS label only.")
    lines.append("")
    lines.append("Shadow-only component: learned objective beta.")
    lines.append("")
    lines.append("| terrain | rows | shadow_rows | rule_beta_mean | learned_clamped_mean | delta_mean |")
    lines.append("|---|---:|---:|---|---|---|")

    for terrain, s in summary.items():
        rb = s["rule_beta"]
        lc = s["learned_clamped"]
        de = s["delta"]

        def fmt(d):
            return (
                f"m={d['motion']['mean']:.3f}, "
                f"s={d['stability']['mean']:.3f}, "
                f"e={d['energy']['mean']:.3f}"
            )

        lines.append(
            f"| {terrain} | {s['rows']} | {s['shadow_rows']} | "
            f"{fmt(rb)} | {fmt(lc)} | {fmt(de)} |"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Learned beta is logged in shadow-only mode and does not affect the command path.")
    lines.append("- The observed direction is generally motion down, stability/energy up, especially on rough and slope terrain.")
    lines.append("- Before active beta deployment, use bounded simplex projection, rate limiting, and low-alpha blending.")

    out_md.write_text("\n".join(lines) + "\n")

    print("[TRACER] wrote beta shadow report")
    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(out_md),
        "terrains": terrains,
    }, indent=2))


if __name__ == "__main__":
    main()
