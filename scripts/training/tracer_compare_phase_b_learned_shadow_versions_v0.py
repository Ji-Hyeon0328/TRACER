#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def fmt(x):
    if x is None:
        return ""
    return f"{float(x):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    r0 = load_json(args.v0)
    r1 = load_json(args.v1)

    worlds = sorted(set(r0.get("by_world", {})) | set(r1.get("by_world", {})))
    rows = []

    for w in worlds:
        a = r0.get("by_world", {}).get(w, {})
        b = r1.get("by_world", {}).get(w, {})
        rows.append({
            "world": w,
            "v0_beta_v": a.get("mean_beta_velocity"),
            "v1_beta_v": b.get("mean_beta_velocity"),
            "v0_beta_s": a.get("mean_beta_stability"),
            "v1_beta_s": b.get("mean_beta_stability"),
            "v0_beta_e": a.get("mean_beta_energy"),
            "v1_beta_e": b.get("mean_beta_energy"),
            "v0_ram_risk": a.get("mean_ram_future_risk"),
            "v1_ram_risk": b.get("mean_ram_future_risk"),
            "v0_ram_uncertainty": a.get("mean_ram_future_uncertainty"),
            "v1_ram_uncertainty": b.get("mean_ram_future_uncertainty"),
        })

    report = {
        "schema": "phase_b_learned_shadow_version_comparison_v0",
        "v0": args.v0,
        "v1": args.v1,
        "rows": rows,
        "interpretation": {
            "purpose": "Compare learned Objective/RAM shadow outputs before and after context-column mapping fixes.",
            "control_effect": "none"
        }
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = []
    lines.append("# Phase-B Learned Shadow v0 vs v1 Comparison")
    lines.append("")
    lines.append("Both versions are shadow-only and do not affect control.")
    lines.append("")
    lines.append("| world | beta_v v0 | beta_v v1 | beta_s v0 | beta_s v1 | beta_e v0 | beta_e v1 | RAM risk v0 | RAM risk v1 | RAM unc v0 | RAM unc v1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        lines.append(
            f"| {r['world']} | {fmt(r['v0_beta_v'])} | {fmt(r['v1_beta_v'])} | "
            f"{fmt(r['v0_beta_s'])} | {fmt(r['v1_beta_s'])} | "
            f"{fmt(r['v0_beta_e'])} | {fmt(r['v1_beta_e'])} | "
            f"{fmt(r['v0_ram_risk'])} | {fmt(r['v1_ram_risk'])} | "
            f"{fmt(r['v0_ram_uncertainty'])} | {fmt(r['v1_ram_uncertainty'])} |"
        )

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
