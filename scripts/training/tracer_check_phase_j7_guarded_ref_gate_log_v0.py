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

reason_counts = Counter(r.get("j7_reason", "unknown") or "unknown" for r in rows)
source_counts = Counter(r.get("j7_source", "unknown") or "unknown" for r in rows)
ctx_counts = Counter(r.get("context", "unknown") or "unknown" for r in rows)

ctx_accept = defaultdict(lambda: {"n": 0, "accepted": 0})
ctx_action_accept = defaultdict(lambda: {"n": 0, "accepted": 0})

for r in rows:
    ctx = r.get("context", "unknown") or "unknown"
    aid = r.get("action_id", "NA") or "NA"
    acc = 1 if r.get("j7_accept") == "1" else 0
    ctx_accept[ctx]["n"] += 1
    ctx_accept[ctx]["accepted"] += acc
    ctx_action_accept[(ctx, aid)]["n"] += 1
    ctx_action_accept[(ctx, aid)]["accepted"] += acc

cols = [
    "emp_vx", "proj_vx", "out_vx", "delta_vx",
    "emp_yaw_rate", "proj_yaw_rate", "out_yaw_rate", "delta_yaw_rate",
    "emp_body_h", "proj_body_h", "out_body_h", "delta_body_h",
    "emp_clearance", "proj_clearance", "out_clearance", "delta_clearance",
    "emp_age_s", "proj_age_s", "theta_age_s",
]

summary_all = {c: stats([ff(r.get(c)) for r in rows]) for c in cols}
summary_acc = {c: stats([ff(r.get(c)) for r in rows if r.get("j7_accept") == "1"]) for c in cols}

viol = {
    "out_vx_low": 0,
    "out_vx_high": 0,
    "out_yaw_low": 0,
    "out_yaw_high": 0,
    "out_body_h_low": 0,
    "out_body_h_high": 0,
    "out_clearance_low": 0,
    "out_clearance_high": 0,
}

for r in rows:
    vx = ff(r.get("out_vx"))
    yaw = ff(r.get("out_yaw_rate"))
    bh = ff(r.get("out_body_h"))
    cl = ff(r.get("out_clearance"))
    if vx is not None:
        viol["out_vx_low"] += int(vx < 0.025 - 1e-9)
        viol["out_vx_high"] += int(vx > 0.220 + 1e-9)
    if yaw is not None:
        viol["out_yaw_low"] += int(yaw < -0.200 - 1e-9)
        viol["out_yaw_high"] += int(yaw > 0.200 + 1e-9)
    if bh is not None:
        viol["out_body_h_low"] += int(bh < 0.300 - 1e-9)
        viol["out_body_h_high"] += int(bh > 0.340 + 1e-9)
    if cl is not None:
        viol["out_clearance_low"] += int(cl < 0.035 - 1e-9)
        viol["out_clearance_high"] += int(cl > 0.065 + 1e-9)

accepted_total = sum(1 for r in rows if r.get("j7_accept") == "1")
accept_rate = accepted_total / len(rows)

lines = []
lines.append("# TRACER Phase-J7 Guarded Ref Gate Log Check v0")
lines.append("")
lines.append(f"- csv: `{p}`")
lines.append(f"- rows: `{len(rows)}`")
lines.append(f"- accepted rows: `{accepted_total}`")
lines.append(f"- accept rate: `{accept_rate:.6f}`")
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
lines.append("## Decision reasons")
lines.append("")
lines.append("| reason | rows |")
lines.append("|---|---:|")
for k, v in reason_counts.most_common():
    lines.append(f"| {k} | {v} |")

lines.append("")
lines.append("## Output source")
lines.append("")
lines.append("| source | rows |")
lines.append("|---|---:|")
for k, v in source_counts.most_common():
    lines.append(f"| {k} | {v} |")

lines.append("")
lines.append("## Acceptance by context")
lines.append("")
lines.append("| context | rows | accepted | accept rate |")
lines.append("|---|---:|---:|---:|")
for ctx in sorted(ctx_accept):
    n = ctx_accept[ctx]["n"]
    a = ctx_accept[ctx]["accepted"]
    lines.append(f"| {ctx} | {n} | {a} | {a / n if n else 0.0:.6f} |")

lines.append("")
lines.append("## Acceptance by context/action")
lines.append("")
lines.append("| context | action_id | rows | accepted | accept rate |")
lines.append("|---|---:|---:|---:|---:|")
for (ctx, aid), d in sorted(ctx_action_accept.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    n = d["n"]
    a = d["accepted"]
    lines.append(f"| {ctx} | {aid} | {n} | {a} | {a / n if n else 0.0:.6f} |")

lines.append("")
lines.append("## Numeric summary: all vs accepted")
lines.append("")
lines.append("| column | all mean | all min | all max | accepted mean | accepted min | accepted max |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for c in cols:
    sa = summary_all[c]
    sg = summary_acc[c]
    lines.append(
        f"| {c} | {fmt(sa['mean'])} | {fmt(sa['min'])} | {fmt(sa['max'])} | "
        f"{fmt(sg['mean'])} | {fmt(sg['min'])} | {fmt(sg['max'])} |"
    )

lines.append("")
lines.append("## Output safety violations")
lines.append("")
lines.append("| check | count |")
lines.append("|---|---:|")
for k, v in viol.items():
    lines.append(f"| {k} | {v} |")

lines.append("")
lines.append("## Safe interpretation")
lines.append("")
lines.append("- J7 is still shadow-only if output topic is `/tracer/meta_action_guarded_ref_shadow`.")
lines.append("- Expected first active candidates should be accepted mainly in flat and upslope.")
lines.append("- Rejected rows should fall back to empirical references.")
lines.append("- Do not route this node to `/tracer/mpc_reference` until the shadow gate report is checked.")

text = "\n".join(lines)
print(text)

if args.out_md:
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"[TRACER] wrote {out}")
