#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, pstdev


def ff(x):
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def fmt(v):
    return "NA" if v is None else f"{v:.6f}"


ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out-md", default="")
args = ap.parse_args()

p = Path(args.csv)
if not p.exists():
    raise SystemExit(f"[TRACER] missing csv: {p}")

rows = list(csv.DictReader(open(p)))
if not rows:
    raise SystemExit(f"[TRACER] empty csv: {p}")

ctx_counts = Counter(r.get("context", "unknown") or "unknown" for r in rows)
action_counts = Counter(r.get("action_id", "NA") for r in rows)
ctx_action = defaultdict(Counter)
for r in rows:
    ctx_action[r.get("context", "unknown") or "unknown"][r.get("action_id", "NA")] += 1

cols = [
    "emp_vx", "proj_vx", "delta_vx",
    "emp_yaw_rate", "proj_yaw_rate", "delta_yaw_rate",
    "emp_body_h", "proj_body_h", "delta_body_h",
    "emp_clearance", "proj_clearance", "delta_clearance",
    "proj_enable",
    "hold_guard",
]

summary = {c: stats([ff(r.get(c)) for r in rows]) for c in cols}

viol = {
    "vx_low": 0,
    "vx_high": 0,
    "yaw_low": 0,
    "yaw_high": 0,
    "body_h_low": 0,
    "body_h_high": 0,
    "clearance_low": 0,
    "clearance_high": 0,
}
for r in rows:
    vx = ff(r.get("proj_vx"))
    yaw = ff(r.get("proj_yaw_rate"))
    bh = ff(r.get("proj_body_h"))
    cl = ff(r.get("proj_clearance"))
    if vx is not None:
        viol["vx_low"] += int(vx < 0.025 - 1e-9)
        viol["vx_high"] += int(vx > 0.220 + 1e-9)
    if yaw is not None:
        viol["yaw_low"] += int(yaw < -0.200 - 1e-9)
        viol["yaw_high"] += int(yaw > 0.200 + 1e-9)
    if bh is not None:
        viol["body_h_low"] += int(bh < 0.300 - 1e-9)
        viol["body_h_high"] += int(bh > 0.340 + 1e-9)
    if cl is not None:
        viol["clearance_low"] += int(cl < 0.035 - 1e-9)
        viol["clearance_high"] += int(cl > 0.065 + 1e-9)

lines = []
lines.append("# TRACER Phase-J4 Meta-Action Ref Projection Shadow Log Check v0")
lines.append("")
lines.append(f"- csv: `{p}`")
lines.append(f"- rows: `{len(rows)}`")
lines.append(f"- first_t: `{rows[0].get('t_wall', '')}`")
lines.append(f"- last_t: `{rows[-1].get('t_wall', '')}`")
lines.append("")

lines.append("## Context counts")
lines.append("")
lines.append("| context | rows |")
lines.append("|---|---:|")
for k, v in sorted(ctx_counts.items()):
    lines.append(f"| {k} | {v} |")

lines.append("")
lines.append("## Action id distribution")
lines.append("")
lines.append("| action_id | rows |")
lines.append("|---:|---:|")
for k, v in action_counts.most_common():
    lines.append(f"| {k} | {v} |")

lines.append("")
lines.append("## Action id by context")
lines.append("")
lines.append("| context | action_id | rows |")
lines.append("|---|---:|---:|")
for ctx in sorted(ctx_action):
    for aid, cnt in ctx_action[ctx].most_common():
        lines.append(f"| {ctx} | {aid} | {cnt} |")

lines.append("")
lines.append("## Numeric summary")
lines.append("")
lines.append("| column | n | mean | std | min | max |")
lines.append("|---|---:|---:|---:|---:|---:|")
for c in cols:
    s = summary[c]
    lines.append(f"| {c} | {s['n']} | {fmt(s['mean'])} | {fmt(s['std'])} | {fmt(s['min'])} | {fmt(s['max'])} |")

lines.append("")
lines.append("## Safety projection violations")
lines.append("")
lines.append("| check | count |")
lines.append("|---|---:|")
for k, v in viol.items():
    lines.append(f"| {k} | {v} |")

lines.append("")
lines.append("## Quick interpretation")
lines.append("")
lines.append("- J4 is shadow-only and should not modify `/tracer/mpc_reference`.")
lines.append("- Expected: projected refs remain inside safety bounds.")
lines.append("- Expected: flat action increases vx slightly, rough/downslope reduce vx or add stability/clearance, goal_flat uses hold guard.")

text = "\n".join(lines)
print(text)

if args.out_md:
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"[TRACER] wrote {out}")
