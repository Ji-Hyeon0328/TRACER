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


def fmt(v):
    return "NA" if v is None else f"{v:.6f}"


ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="logs/phase_h/h2_objective_shadow_latest/objective_selector_h2_shadow_v0.csv")
ap.add_argument("--out-md", default="")
args = ap.parse_args()

p = Path(args.csv)
if not p.exists():
    raise SystemExit(f"[TRACER] missing csv: {p}")

rows = list(csv.DictReader(open(p)))
if not rows:
    raise SystemExit(f"[TRACER] empty csv: {p}")

cols = [
    "beta_motion",
    "beta_stability",
    "beta_energy",
    "observed_features",
    "imputed_features",
    "x",
    "y",
    "ref_vx",
    "ram_slip_proxy",
    "ram_roughness_proxy",
    "ram_sigma",
]

summary = {c: stats([ff(r.get(c)) for r in rows]) for c in cols}

contexts = {}
for r in rows:
    contexts[r.get("context", "unknown")] = contexts.get(r.get("context", "unknown"), 0) + 1

lines = []
lines.append("# TRACER Phase-H2 Objective Selector Shadow Log Check v0")
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
    lines.append(
        f"| {c} | {s['n']} | {fmt(s['mean'])} | {fmt(s['std'])} | {fmt(s['min'])} | {fmt(s['max'])} |"
    )

lines.append("")
lines.append("## Quick interpretation")
lines.append("")
obs = summary["observed_features"]["mean"]
imp = summary["imputed_features"]["mean"]
if obs is not None and imp is not None:
    lines.append(f"- average observed/imputed features: `{obs:.2f}` / `{imp:.2f}`")
    if imp > obs:
        lines.append("- Many features are still imputed. Treat H2 as shadow diagnostic, not active Objective Selector.")
    else:
        lines.append("- Most features are observed from runtime/window signals.")
lines.append("- H2 beta should not control D6/D7 yet unless compared against rollout behavior.")

text = "\n".join(lines)
print(text)

if args.out_md:
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"[TRACER] wrote {out}")
