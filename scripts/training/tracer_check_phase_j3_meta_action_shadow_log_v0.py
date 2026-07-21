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
act_counts = Counter(r.get("action_name", "unknown") or "unknown" for r in rows)
reason_counts = Counter(r.get("reason", "unknown") or "unknown" for r in rows)

ctx_act = defaultdict(Counter)
for r in rows:
    ctx_act[r.get("context", "unknown") or "unknown"][r.get("action_name", "unknown") or "unknown"] += 1

cols = [
    "x", "y",
    "beta_motion", "beta_stability", "beta_energy",
    "ram_slip", "ram_rough", "ram_sigma",
    "vx_scale", "body_h_delta", "clearance_delta",
    "stability_bias", "energy_bias",
]

summary = {c: stats([ff(r.get(c)) for r in rows]) for c in cols}

lines = []
lines.append("# TRACER Phase-J3 Meta-Action Shadow Log Check v0")
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
lines.append("## Action distribution")
lines.append("")
lines.append("| action | rows |")
lines.append("|---|---:|")
for k, v in act_counts.most_common():
    lines.append(f"| {k} | {v} |")

lines.append("")
lines.append("## Action by context")
lines.append("")
lines.append("| context | action | rows |")
lines.append("|---|---|---:|")
for ctx in sorted(ctx_act):
    for action_name, cnt in ctx_act[ctx].most_common():
        lines.append(f"| {ctx} | {action_name} | {cnt} |")

lines.append("")
lines.append("## Selection reasons")
lines.append("")
lines.append("| reason | rows |")
lines.append("|---|---:|")
for k, v in reason_counts.most_common():
    lines.append(f"| {k} | {v} |")

lines.append("")
lines.append("## Numeric summary")
lines.append("")
lines.append("| column | n | mean | std | min | max |")
lines.append("|---|---:|---:|---:|---:|---:|")
for c in cols:
    s = summary[c]
    lines.append(f"| {c} | {s['n']} | {fmt(s['mean'])} | {fmt(s['std'])} | {fmt(s['min'])} | {fmt(s['max'])} |")

lines.append("")
lines.append("## Quick interpretation")
lines.append("")
lines.append("- J3 is shadow-only and should not modify `/tracer/mpc_reference`.")
lines.append("- Expected mapping: flat->fast_motion, upslope->upslope_push, downslope->downslope_stable, rough->lateral_recovery_soft/rough_stability, goal_flat->goal_hold.")
lines.append("- If context/action transitions are visible, the meta-action policy runtime path is connected.")

text = "\n".join(lines)
print(text)

if args.out_md:
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"[TRACER] wrote {out}")
