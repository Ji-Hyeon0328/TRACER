#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean, pstdev


TERRAINS = ["flat_normal", "rough_mid", "slope_5deg"]
KEYS = ["motion", "stability", "energy"]


def load_jsonl(path: Path):
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


def load_latest_summary(terrain: str):
    pattern = (
        "data/rollout_dataset_v0/summaries/"
        f"{terrain}_theta_mlp_udp_v0_learned_stack_v3_beta_blend_on_{terrain}_r*.json"
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        # flat terrain has sometimes been logged as flat_normal in policy_id suffix.
        pattern = (
            "data/rollout_dataset_v0/summaries/"
            f"{terrain}_theta_mlp_udp_v0_learned_stack_v3_beta_blend_on_*_r*.json"
        )
        paths = sorted(glob.glob(pattern))

    if not paths:
        return None

    p = Path(paths[-1])
    try:
        d = json.loads(p.read_text())
        d["_summary_path"] = str(p)
        return d
    except Exception:
        return None


def stat(xs):
    if not xs:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(xs),
        "std": pstdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def fmt_num(x, nd=4):
    if x is None:
        return "n/a"
    return f"{float(x):.{nd}f}"


def fmt_beta_mean(d):
    return (
        f"m={fmt_num(d['motion']['mean'], 3)}, "
        f"s={fmt_num(d['stability']['mean'], 3)}, "
        f"e={fmt_num(d['energy']['mean'], 3)}"
    )


def summarize_debug(terrain: str, path: Path):
    rows = load_jsonl(path)

    reasons = {}
    applied = []

    for r in rows:
        bb = r.get("learned_stack_v3_beta_blend") or {}
        reason = str(bb.get("reason"))
        reasons[reason] = reasons.get(reason, 0) + 1
        if bb.get("applied"):
            applied.append(bb)

    out = {
        "terrain": terrain,
        "debug_path": str(path),
        "rows": len(rows),
        "applied_rows": len(applied),
        "reasons": reasons,
        "rule_beta": {},
        "learned_beta": {},
        "blended_beta": {},
        "delta_from_rule": {},
    }

    field_map = {
        "rule_beta": "rule_beta",
        "learned_beta_clamped": "learned_beta",
        "blended_beta": "blended_beta",
        "delta_from_rule": "delta_from_rule",
    }

    for src_field, dst_field in field_map.items():
        for k in KEYS:
            xs = []
            for bb in applied:
                d = bb.get(src_field) or {}
                if k in d:
                    xs.append(float(d[k]))
            out[dst_field][k] = stat(xs)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prefix",
        default="/tmp/tracer_highlevel_debug_beta_blend_on",
        help="Input prefix for debug captures. Expected: {prefix}_{terrain}.txt",
    )
    ap.add_argument(
        "--out-json",
        default="reports/learned_stack_v3_beta_blend_report.json",
    )
    ap.add_argument(
        "--out-md",
        default="reports/learned_stack_v3_beta_blend_report.md",
    )
    args = ap.parse_args()

    report = {
        "version": "learned_stack_v3_beta_blend_v0_report",
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
        "terrains": {},
    }

    for terrain in TERRAINS:
        debug_path = Path(f"{args.prefix}_{terrain}.txt")
        debug_summary = summarize_debug(terrain, debug_path)
        rollout_summary = load_latest_summary(terrain)

        report["terrains"][terrain] = {
            "debug": debug_summary,
            "rollout": rollout_summary,
        }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2))

    lines = []
    lines.append("# TRACER Learned Stack v3 Beta Blend v0 Report")
    lines.append("")
    lines.append("Active components:")
    lines.append("- learned GMS label")
    lines.append("- learned objective beta low-alpha blend")
    lines.append("")
    lines.append("Safety:")
    lines.append("- alpha-limited beta blend")
    lines.append("- terrain-specific max delta limit")
    lines.append("- GMS-only safety path maintained")
    lines.append("- RAM-triggered recovery remains inactive")
    lines.append("")
    lines.append("| terrain | rows | applied | delta_mean | blended_beta_mean | vx_mean | success |")
    lines.append("|---|---:|---:|---|---|---:|---:|")

    for terrain in TERRAINS:
        item = report["terrains"][terrain]
        dbg = item["debug"]
        roll = item["rollout"] or {}

        delta = dbg["delta_from_rule"]
        blend = dbg["blended_beta"]

        vx = roll.get("mpc_vx_mean")
        success = roll.get("success_proxy")

        lines.append(
            f"| {terrain} | "
            f"{dbg['rows']} | "
            f"{dbg['applied_rows']} | "
            f"{fmt_beta_mean(delta)} | "
            f"{fmt_beta_mean(blend)} | "
            f"{fmt_num(vx, 4)} | "
            f"{success} |"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Beta blend v0 applies a low-alpha learned beta adjustment while preserving the GMS-only safety path.")
    lines.append("- Delta from rule beta should remain small on all terrains.")
    lines.append("- Rollout success should remain true before moving to robust repeated evaluation.")
    lines.append("- This report does not validate learned RAM as an active recovery gate.")

    out_md.write_text("\n".join(lines) + "\n")

    print("[TRACER] wrote beta blend report")
    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(out_md),
        "terrains": TERRAINS,
    }, indent=2))


if __name__ == "__main__":
    main()
