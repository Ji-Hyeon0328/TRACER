#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

LEGS = ["FL", "FR", "RL", "RR"]

def ff(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.in_jsonl, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    out_rows = []

    by_action = defaultdict(lambda: {
        "n": 0,
        "hard_invalid": 0,
        "warning": 0,
        "missing_state": 0,
        "hard_reasons": defaultdict(int),
        "warning_reasons": defaultdict(int),
    })

    by_world = defaultdict(lambda: {
        "n": 0,
        "hard_invalid": 0,
        "warning": 0,
        "missing_state": 0,
    })

    for r in rows:
        f = r.get("features", {})
        hard = []
        warn = []

        has_state = bool(r.get("has_state", False))
        world = r.get("world", "unknown")
        action = r.get("action", "unknown")
        key = f"{world}::{action}"

        if not has_state:
            hard.append("missing_state_columns")

        mean_base_z = ff(f.get("mean_base_z"), 999.0)
        min_base_z = ff(f.get("min_base_z"), 999.0)
        max_abs_roll = ff(f.get("max_abs_roll"), 0.0)
        max_abs_pitch = ff(f.get("max_abs_pitch"), 0.0)

        # Hard collapse: clearly too low.
        if mean_base_z < 0.16 or min_base_z < 0.10:
            hard.append("severe_low_base_height")
        # Soft warning: sponge sink / crouch-like low posture.
        elif mean_base_z < 0.22 or min_base_z < 0.16:
            warn.append("soft_low_base_height")

        if max_abs_roll > 0.80 or max_abs_pitch > 0.80:
            hard.append("large_roll_or_pitch")

        folded = f.get("folded_frac", {}) or {}
        folded_legs = [leg for leg in LEGS if ff(folded.get(leg), 0.0) > 0.50]
        if folded_legs:
            hard.append("folded_joint_posture:" + ",".join(folded_legs))

        if ff(f.get("diag_motion_ratio"), 0.0) > 0.55:
            hard.append("diagonal_motion_imbalance")

        if ff(f.get("diag_contact_ratio"), 0.0) > 0.55:
            hard.append("diagonal_contact_imbalance")

        if ff(f.get("motion_spread_ratio"), 0.0) > 0.75:
            hard.append("per_leg_motion_collapse")

        if ff(f.get("contact_spread_ratio"), 0.0) > 0.65:
            hard.append("contact_asymmetry")

        # Metric-only suspicious should be warning unless physical hard reason exists.
        reached = bool(f.get("reached_stop_distance", False))
        final_d = ff(f.get("final_rel_dist"), 999.0)
        progress = ff(f.get("progress_initial_minus_min"), 0.0)
        if (not reached) and final_d < 1.2 and progress < 0.25:
            warn.append("metric_suspicious_low_progress_stable")

        hard_invalid = len(hard) > 0
        warning = len(warn) > 0

        nr = dict(r)
        nr["schema"] = "phase_b_invalid_gait_detection_v1b_calibrated"
        nr["hard_invalid_gait"] = hard_invalid
        nr["warning_gait"] = warning
        nr["hard_reasons"] = hard
        nr["warning_reasons"] = warn
        out_rows.append(nr)

        by_action[key]["n"] += 1
        by_world[world]["n"] += 1

        if not has_state:
            by_action[key]["missing_state"] += 1
            by_world[world]["missing_state"] += 1

        if hard_invalid:
            by_action[key]["hard_invalid"] += 1
            by_world[world]["hard_invalid"] += 1
            for reason in hard:
                by_action[key]["hard_reasons"][reason] += 1

        if warning:
            by_action[key]["warning"] += 1
            by_world[world]["warning"] += 1
            for reason in warn:
                by_action[key]["warning_reasons"][reason] += 1

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    report = {
        "schema": "phase_b_invalid_gait_calibrated_report_v1b",
        "input_jsonl": args.in_jsonl,
        "num_episodes": len(out_rows),
        "num_hard_invalid": sum(1 for r in out_rows if r["hard_invalid_gait"]),
        "num_warning": sum(1 for r in out_rows if r["warning_gait"]),
        "by_world": {},
        "by_action": {},
    }

    for k, v in sorted(by_world.items()):
        n = max(1, v["n"])
        report["by_world"][k] = {
            "n": v["n"],
            "hard_invalid": v["hard_invalid"],
            "warning": v["warning"],
            "missing_state": v["missing_state"],
            "hard_invalid_rate": v["hard_invalid"] / n,
            "warning_rate": v["warning"] / n,
            "missing_state_rate": v["missing_state"] / n,
        }

    for k, v in sorted(by_action.items()):
        n = max(1, v["n"])
        report["by_action"][k] = {
            "n": v["n"],
            "hard_invalid": v["hard_invalid"],
            "warning": v["warning"],
            "missing_state": v["missing_state"],
            "hard_invalid_rate": v["hard_invalid"] / n,
            "warning_rate": v["warning"] / n,
            "missing_state_rate": v["missing_state"] / n,
            "top_hard_reasons": sorted(v["hard_reasons"].items(), key=lambda x: (-x[1], x[0]))[:8],
            "top_warning_reasons": sorted(v["warning_reasons"].items(), key=lambda x: (-x[1], x[0]))[:8],
        }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = []
    lines.append("# Phase-B Invalid Gait Calibration v1b")
    lines.append("")
    lines.append(f"- Input: `{args.in_jsonl}`")
    lines.append(f"- Episodes: `{report['num_episodes']}`")
    lines.append(f"- Hard invalid: `{report['num_hard_invalid']}`")
    lines.append(f"- Warning: `{report['num_warning']}`")
    lines.append("")
    lines.append("## By world")
    lines.append("")
    lines.append("| world | n | hard_invalid_rate | warning_rate | missing_state_rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for k, v in report["by_world"].items():
        lines.append(
            f"| {k} | {v['n']} | {v['hard_invalid_rate']:.3f} | "
            f"{v['warning_rate']:.3f} | {v['missing_state_rate']:.3f} |"
        )

    lines.append("")
    lines.append("## By action")
    lines.append("")
    lines.append("| world::action | n | hard_invalid_rate | warning_rate | hard reasons | warning reasons |")
    lines.append("|---|---:|---:|---:|---|---|")
    for k, v in report["by_action"].items():
        hard_r = ", ".join(f"{name}:{cnt}" for name, cnt in v["top_hard_reasons"])
        warn_r = ", ".join(f"{name}:{cnt}" for name, cnt in v["top_warning_reasons"])
        lines.append(
            f"| {k} | {v['n']} | {v['hard_invalid_rate']:.3f} | "
            f"{v['warning_rate']:.3f} | {hard_r} | {warn_r} |"
        )

    md = "\n".join(lines) + "\n"
    Path(args.out_md).write_text(md)

    print(md)
    print("[wrote]", args.out_jsonl)
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)

if __name__ == "__main__":
    main()
