#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict


CONTEXTS = ["flat", "start_flat", "upslope", "rough", "downslope", "goal_flat", "unknown"]


def fnum(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if len(xs) <= 1:
        return 0.0 if xs else None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--min-moving-vx", type=float, default=0.10)
    ap.add_argument("--hold-vx-threshold", type=float, default=0.10)
    ap.add_argument("--yaw-max", type=float, default=0.20)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    out_json = Path(args.out_json)
    out_summary = Path(args.out_summary)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(dataset.open()))

    moving_by_ctx = defaultdict(list)
    hold_rows = []
    yaw_pairs = []

    for r in rows:
        ctx = r.get("context", "unknown")
        if ctx not in CONTEXTS:
            ctx = "unknown"

        vx = fnum(r.get("target_vx"))
        yaw = fnum(r.get("target_yaw_rate"))
        h = fnum(r.get("target_body_h"))
        clr = fnum(r.get("target_clearance"))
        y = fnum(r.get("y"))
        enable = fnum(r.get("target_enable"), 1.0)

        if vx is None or h is None or clr is None or enable is None:
            continue

        if enable <= 0.5:
            continue

        if vx >= args.min_moving_vx:
            moving_by_ctx[ctx].append({
                "vx": vx,
                "yaw": yaw,
                "body_h": h,
                "clearance": clr,
                "y": y,
            })
            if y is not None and yaw is not None and abs(y) > 1e-4:
                yaw_pairs.append((y, yaw))

        if vx < args.hold_vx_threshold:
            hold_rows.append({
                "vx": vx,
                "yaw": yaw,
                "body_h": h,
                "clearance": clr,
                "y": y,
            })

    # Fill context action table from moving rows.
    context_stats = {}
    fallback = {
        "vx": 0.2025,
        "body_h": 0.320,
        "clearance": 0.045,
    }

    for ctx in CONTEXTS:
        rs = moving_by_ctx.get(ctx, [])
        if rs:
            vx_m = mean([r["vx"] for r in rs])
            h_m = mean([r["body_h"] for r in rs])
            c_m = mean([r["clearance"] for r in rs])
            context_stats[ctx] = {
                "n": len(rs),
                "vx_mean": vx_m,
                "vx_std": std([r["vx"] for r in rs]),
                "body_h_mean": h_m,
                "clearance_mean": c_m,
            }
        else:
            context_stats[ctx] = {
                "n": 0,
                "vx_mean": fallback["vx"],
                "vx_std": None,
                "body_h_mean": fallback["body_h"],
                "clearance_mean": fallback["clearance"],
            }

    # start_flat was not present in the course dataset; tie it to flat.
    if context_stats["start_flat"]["n"] == 0 and context_stats["flat"]["n"] > 0:
        context_stats["start_flat"] = dict(context_stats["flat"])
        context_stats["start_flat"]["copied_from"] = "flat"

    if context_stats["unknown"]["n"] == 0 and context_stats["goal_flat"]["n"] > 0:
        context_stats["unknown"] = dict(context_stats["goal_flat"])
        context_stats["unknown"]["copied_from"] = "goal_flat"

    hold_vx = mean([r["vx"] for r in hold_rows])
    if hold_vx is None:
        hold_vx = 0.025

    # Linear least-squares fit yaw_rate ≈ gain * y.
    denom = sum(y * y for y, yaw in yaw_pairs)
    yaw_gain = sum(y * yaw for y, yaw in yaw_pairs) / denom if denom > 1e-12 else -0.025
    yaw_sign = -1.0 if yaw_gain < 0 else 1.0
    yaw_k = abs(yaw_gain)

    # Keep yaw_max as a conservative runtime cap, not the observed max.
    yaw_max = float(args.yaw_max)

    def fmt_action(ctx):
        st = context_stats[ctx]
        return f"{ctx}:{st['vx_mean']:.4f},{st['body_h_mean']:.3f},{st['clearance_mean']:.3f}"

    action_table = ";".join(fmt_action(ctx) for ctx in CONTEXTS)

    policy = {
        "policy_name": "phase_d5_empirical_context_beta_ram_shadow_v0",
        "policy_type": "empirical_behavior_cloning_table",
        "source_dataset": str(dataset),
        "note": (
            "Exported from Phase-D5 shadow logs. This is a dataset-derived "
            "empirical policy artifact, not yet a neural RL policy."
        ),
        "action_table": action_table,
        "hold_vx": round(float(hold_vx), 5),
        "yaw_correction": {
            "type": "proportional_lateral_feedback",
            "yaw_gain": round(float(yaw_gain), 6),
            "yaw_sign": float(yaw_sign),
            "yaw_k": round(float(yaw_k), 6),
            "yaw_max": yaw_max,
            "fit_method": "least_squares target_yaw_rate = yaw_gain * y over moving rows",
            "num_fit_samples": len(yaw_pairs),
        },
        "context_stats": context_stats,
        "hold_stats": {
            "n": len(hold_rows),
            "hold_vx_mean": hold_vx,
            "hold_vx_std": std([r["vx"] for r in hold_rows]),
        },
    }

    out_json.write_text(json.dumps(policy, indent=2))

    lines = []
    lines.append("# TRACER Phase-D5 Empirical Policy Export v0")
    lines.append("")
    lines.append(f"- source_dataset: `{dataset}`")
    lines.append(f"- out_json: `{out_json}`")
    lines.append(f"- hold_vx: `{policy['hold_vx']}`")
    lines.append(f"- yaw_gain: `{policy['yaw_correction']['yaw_gain']}`")
    lines.append(f"- yaw_k: `{policy['yaw_correction']['yaw_k']}`")
    lines.append(f"- yaw_sign: `{policy['yaw_correction']['yaw_sign']}`")
    lines.append(f"- yaw_max: `{policy['yaw_correction']['yaw_max']}`")
    lines.append("")
    lines.append("## Context action table")
    lines.append("")
    lines.append("| context | n | vx_mean | vx_std | body_h | clearance |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for ctx in CONTEXTS:
        st = context_stats[ctx]
        vx_std_text = "" if st["vx_std"] is None else f"{st['vx_std']:.6f}"
        lines.append(
            f"| {ctx} | {st.get('n', 0)} | "
            f"{st['vx_mean']:.4f} | "
            f"{vx_std_text} | "
            f"{st['body_h_mean']:.3f} | {st['clearance_mean']:.3f} |"
        )
    lines.append("")
    lines.append("## Action table string")
    lines.append("")
    lines.append("```text")
    lines.append(action_table)
    lines.append("```")

    out_summary.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] wrote {out_json}")
    print(f"[TRACER] wrote {out_summary}")
    print(f"[TRACER] hold_vx={policy['hold_vx']}")
    print(f"[TRACER] yaw_gain={policy['yaw_correction']['yaw_gain']}")
    print(f"[TRACER] action_table={action_table}")


if __name__ == "__main__":
    main()
