#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
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


def sat_rate(vals, lo=0.02, hi=0.98):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v <= lo or v >= hi) / len(vals)


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

contexts = {}
for r in rows:
    contexts[r.get("context", "unknown")] = contexts.get(r.get("context", "unknown"), 0) + 1

cols = [
    "safe_slip", "safe_rough", "safe_sigma",
    "learned_slip", "learned_rough", "learned_sigma",
    "damped_slip", "damped_rough", "damped_sigma",
    "legacy_slip", "legacy_rough", "legacy_sigma",
    "learned_l1_legacy", "damped_l1_legacy", "safe_l1_legacy",
    "used_legacy", "x", "y",
]

summary = {c: stats([ff(r.get(c)) for r in rows]) for c in cols}

lines = []
lines.append("# TRACER Phase-I5 Safe RAM Shadow Log Check v0")
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
for k in sorted(contexts):
    lines.append(f"| {k} | {contexts[k]} |")

lines.append("")
lines.append("## Numeric summary")
lines.append("")
lines.append("| column | n | mean | std | min | max |")
lines.append("|---|---:|---:|---:|---:|---:|")
for c in cols:
    s = summary[c]
    lines.append(f"| {c} | {s['n']} | {fmt(s['mean'])} | {fmt(s['std'])} | {fmt(s['min'])} | {fmt(s['max'])} |")

lines.append("")
lines.append("## Saturation")
lines.append("")
lines.append("| output | slip | rough | sigma |")
lines.append("|---|---:|---:|---:|")
for prefix in ["learned", "damped", "safe"]:
    lines.append(
        f"| {prefix} | "
        f"{fmt(sat_rate([ff(r.get(prefix + '_slip')) for r in rows]))} | "
        f"{fmt(sat_rate([ff(r.get(prefix + '_rough')) for r in rows]))} | "
        f"{fmt(sat_rate([ff(r.get(prefix + '_sigma')) for r in rows]))} |"
    )

lines.append("")
lines.append("## Quick interpretation")
lines.append("")
safe_l1 = summary["safe_l1_legacy"]["mean"]
learned_l1 = summary["learned_l1_legacy"]["mean"]
if safe_l1 is not None and learned_l1 is not None:
    lines.append(f"- learned L1 vs legacy: `{learned_l1:.6f}`")
    lines.append(f"- safe L1 vs legacy: `{safe_l1:.6f}`")
    if safe_l1 < learned_l1:
        lines.append("- I5 safe postprocess reduces distance from legacy RAM while preserving learned RAM direction.")
lines.append("- I5 is still a shadow output. Do not replace `/tracer/ram_mismatch` yet.")

text = "\n".join(lines)
print(text)

if args.out_md:
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"[TRACER] wrote {out}")
