#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def load_json(p):
    with Path(p).open() as f:
        return json.load(f)


def infer_tag(path):
    name = Path(path).name
    for tag in ["m060", "m030", "p000", "p030", "p060"]:
        if f"yoffset_{tag}_n3" in name:
            return tag
    return "unknown"


def tag_to_offset(tag):
    return {
        "m060": "-0.60",
        "m030": "-0.30",
        "p000": "0.00",
        "p030": "0.30",
        "p060": "0.60",
    }.get(tag, "unknown")


def fmt(x):
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.6f}"
    return str(x)


def find_manifest_from_summary(summary):
    return summary.get("manifest") or summary.get("manifest_path") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    entries = []
    for p in args.summaries:
        js = load_json(p)
        tag = infer_tag(p)
        entries.append((tag, tag_to_offset(tag), Path(p), js))

    entries.sort(key=lambda x: {"m060": 0, "m030": 1, "p000": 2, "p030": 3, "p060": 4}.get(x[0], 99))

    lines = []
    lines.append("# TRACER Phase-D6.3b Lateral Offset Robustness Summary v0")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This report aggregates Phase-D6 MLP-gated selector control under lateral reset offsets.")
    lines.append("")
    lines.append("The validated runtime structure is:")
    lines.append("")
    lines.append("```text")
    lines.append("MLP selector -> gate -> /tracer/mpc_reference")
    lines.append("")
    lines.append("learned selected dimensions:")
    lines.append("  vx=True")
    lines.append("  yaw=True")
    lines.append("  clearance=True")
    lines.append("")
    lines.append("empirical/safety protected dimensions:")
    lines.append("  body_h=False")
    lines.append("  enable=False")
    lines.append("  goal/hold phase = empirical fallback")
    lines.append("```")
    lines.append("")
    lines.append("## Offset-level summary")
    lines.append("")
    lines.append("| tag | reset_y | n | success_rate | goal_rate | startup_failed_rate | out_lane_rate | hold_drift_mean | summary_json |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")

    for tag, offset, p, js in entries:
        lines.append(
            f"| {tag} | {offset} | "
            f"{fmt(js.get('n'))} | "
            f"{fmt(js.get('success_rate'))} | "
            f"{fmt(js.get('goal_rate'))} | "
            f"{fmt(js.get('startup_failed_rate'))} | "
            f"{fmt(js.get('out_lane_rate'))} | "
            f"{fmt(js.get('hold_drift_mean'))} | "
            f"`{p}` |"
        )

    lines.append("")
    lines.append("## Individual rollouts")
    lines.append("")
    lines.append("| tag | reset_y | trial | success | goal | startup_failed | out_lane | final_x | final_y | max_abs_y | mean_abs_y | hold_drift | final_context |")
    lines.append("|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---|")

    for tag, offset, p, js in entries:
        rollouts = js.get("rollouts") or js.get("individual_rollouts") or []
        for r in rollouts:
            lines.append(
                f"| {tag} | {offset} | "
                f"{fmt(r.get('trial'))} | "
                f"{fmt(r.get('success'))} | "
                f"{fmt(r.get('goal'))} | "
                f"{fmt(r.get('startup_failed'))} | "
                f"{fmt(r.get('out_lane'))} | "
                f"{fmt(r.get('final_x'))} | "
                f"{fmt(r.get('final_y'))} | "
                f"{fmt(r.get('max_abs_y'))} | "
                f"{fmt(r.get('mean_abs_y'))} | "
                f"{fmt(r.get('hold_drift'))} | "
                f"{fmt(r.get('final_context'))} |"
            )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Phase-D6.3b checks whether the MLP-gated selector remains valid when the robot is laterally perturbed at reset.")
    lines.append("")
    lines.append("A successful result means the controller is not only memorizing the near-center initial condition. It can still reach the goal while remaining inside the lateral bound under moderate initial y-offset perturbations.")
    lines.append("")
    lines.append("This remains a supervised learned selector with a safety gate, not a full RL meta-planner.")
    lines.append("")

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[TRACER] wrote {out}")


if __name__ == "__main__":
    main()
