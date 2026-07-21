#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict
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


def summarize_group(name, rows):
    out = {"group": name, "rows": len(rows)}
    for c in [
        "emp_vx", "proj_vx", "delta_vx",
        "emp_yaw_rate", "proj_yaw_rate", "delta_yaw_rate",
        "emp_body_h", "proj_body_h", "delta_body_h",
        "emp_clearance", "proj_clearance", "delta_clearance",
        "hold_guard",
    ]:
        st = stats([ff(r.get(c)) for r in rows])
        out[f"{c}_mean"] = st["mean"]
        out[f"{c}_std"] = st["std"]
        out[f"{c}_min"] = st["min"]
        out[f"{c}_max"] = st["max"]
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out-csv", default="datasets/phase_j/j5_ref_projection_effect_summary_v0.csv")
ap.add_argument("--out-md", default="reports/phase_j/j5_ref_projection_effect_summary_v0.md")
args = ap.parse_args()

p = Path(args.csv)
if not p.exists():
    raise SystemExit(f"[TRACER] missing csv: {p}")

rows = list(csv.DictReader(open(p)))
if not rows:
    raise SystemExit(f"[TRACER] empty csv: {p}")

groups = []
groups.append(summarize_group("ALL", rows))

by_ctx = defaultdict(list)
by_action = defaultdict(list)
by_ctx_action = defaultdict(list)

for r in rows:
    ctx = r.get("context", "unknown") or "unknown"
    aid = r.get("action_id", "NA") or "NA"
    by_ctx[ctx].append(r)
    by_action[aid].append(r)
    by_ctx_action[(ctx, aid)].append(r)

for ctx in sorted(by_ctx):
    groups.append(summarize_group(f"context:{ctx}", by_ctx[ctx]))

for aid in sorted(by_action, key=lambda x: int(x) if str(x).isdigit() else 999):
    groups.append(summarize_group(f"action:{aid}", by_action[aid]))

for ctx, aid in sorted(by_ctx_action, key=lambda kv: (kv[0][0], int(kv[0][1]) if str(kv[0][1]).isdigit() else 999)):
    groups.append(summarize_group(f"context_action:{ctx}:{aid}", by_ctx_action[(ctx, aid)]))

out_csv = Path(args.out_csv)
out_csv.parent.mkdir(parents=True, exist_ok=True)

fields = [
    "group", "rows",
    "emp_vx_mean", "proj_vx_mean", "delta_vx_mean", "delta_vx_min", "delta_vx_max",
    "emp_yaw_rate_mean", "proj_yaw_rate_mean", "delta_yaw_rate_mean",
    "emp_body_h_mean", "proj_body_h_mean", "delta_body_h_mean",
    "emp_clearance_mean", "proj_clearance_mean", "delta_clearance_mean",
    "hold_guard_mean",
]

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for g in groups:
        w.writerow({k: g.get(k, "") for k in fields})

out_md = Path(args.out_md)
out_md.parent.mkdir(parents=True, exist_ok=True)

with open(out_md, "w") as f:
    f.write("# TRACER Phase-J5 Ref Projection Effect Summary v0\n\n")
    f.write("This summarizes how the J4 shadow meta-action projection changes empirical references by context and action.\n\n")
    f.write(f"- input csv: `{p}`\n")
    f.write(f"- output csv: `{out_csv}`\n")
    f.write(f"- rows: `{len(rows)}`\n\n")

    f.write("## Overall\n\n")
    g = groups[0]
    f.write("| rows | emp vx | proj vx | delta vx | emp clearance | proj clearance | delta clearance | hold rate |\n")
    f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    f.write(
        f"| {g['rows']} | {fmt(g['emp_vx_mean'])} | {fmt(g['proj_vx_mean'])} | {fmt(g['delta_vx_mean'])} | "
        f"{fmt(g['emp_clearance_mean'])} | {fmt(g['proj_clearance_mean'])} | {fmt(g['delta_clearance_mean'])} | "
        f"{fmt(g['hold_guard_mean'])} |\n"
    )

    f.write("\n## By context\n\n")
    f.write("| context | rows | emp vx | proj vx | delta vx | proj body_h | delta body_h | proj clearance | delta clearance | hold rate |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for g in groups:
        if not g["group"].startswith("context:"):
            continue
        ctx = g["group"].replace("context:", "")
        f.write(
            f"| {ctx} | {g['rows']} | {fmt(g['emp_vx_mean'])} | {fmt(g['proj_vx_mean'])} | {fmt(g['delta_vx_mean'])} | "
            f"{fmt(g['proj_body_h_mean'])} | {fmt(g['delta_body_h_mean'])} | "
            f"{fmt(g['proj_clearance_mean'])} | {fmt(g['delta_clearance_mean'])} | {fmt(g['hold_guard_mean'])} |\n"
        )

    f.write("\n## By context/action\n\n")
    f.write("| context | action_id | rows | emp vx | proj vx | delta vx | proj body_h | proj clearance | delta clearance | hold rate |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for g in groups:
        if not g["group"].startswith("context_action:"):
            continue
        _, ctx, aid = g["group"].split(":")
        f.write(
            f"| {ctx} | {aid} | {g['rows']} | {fmt(g['emp_vx_mean'])} | {fmt(g['proj_vx_mean'])} | {fmt(g['delta_vx_mean'])} | "
            f"{fmt(g['proj_body_h_mean'])} | {fmt(g['proj_clearance_mean'])} | {fmt(g['delta_clearance_mean'])} | {fmt(g['hold_guard_mean'])} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("- J5 is an offline summary of J4 shadow projection only.\n")
    f.write("- It does not modify active control.\n")
    f.write("- Good signs: flat slightly increases vx, rough/downslope reduce vx or increase stability margin, goal_flat holds, and all projected refs remain within safety bounds.\n")
    f.write("- If these projected references look reasonable, the next step is not full active control yet, but a guarded active candidate design.\n")

print(f"[TRACER] wrote {out_csv}")
print(f"[TRACER] wrote {out_md}")
print(f"[TRACER] rows={len(rows)}")
