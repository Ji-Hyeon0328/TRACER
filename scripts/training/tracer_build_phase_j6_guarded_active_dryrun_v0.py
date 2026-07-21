#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, pstdev


def ff(x, default=None):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


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


def decide(row, args):
    ctx = row.get("context", "unknown") or "unknown"
    action_id = row.get("action_id", "NA") or "NA"

    emp_vx = ff(row.get("emp_vx"))
    proj_vx = ff(row.get("proj_vx"))
    emp_yaw = ff(row.get("emp_yaw_rate"))
    proj_yaw = ff(row.get("proj_yaw_rate"))
    emp_body_h = ff(row.get("emp_body_h"))
    proj_body_h = ff(row.get("proj_body_h"))
    emp_clearance = ff(row.get("emp_clearance"))
    proj_clearance = ff(row.get("proj_clearance"))
    hold_guard = ff(row.get("hold_guard"), 0.0)

    if None in [emp_vx, proj_vx, emp_yaw, proj_yaw, emp_body_h, proj_body_h, emp_clearance, proj_clearance]:
        return False, "missing_numeric"

    dvx = proj_vx - emp_vx
    dyaw = proj_yaw - emp_yaw
    dbh = proj_body_h - emp_body_h
    dcl = proj_clearance - emp_clearance

    # Never use meta-action projection to override existing goal/hold behavior in the first active candidate.
    if hold_guard >= 0.5 or ctx == "goal_flat" or action_id == "8":
        return False, "hold_or_goal_protected"

    # Basic safety envelope.
    if abs(dvx) > args.max_abs_delta_vx:
        return False, "delta_vx_too_large"
    if abs(dyaw) > args.max_abs_delta_yaw:
        return False, "delta_yaw_too_large"
    if abs(dbh) > args.max_abs_delta_body_h:
        return False, "delta_body_h_too_large"
    if abs(dcl) > args.max_abs_delta_clearance:
        return False, "delta_clearance_too_large"

    if proj_body_h < args.min_active_body_h:
        return False, "body_h_too_low_for_active_candidate"

    # Do not allow aggressive rough/downslope recovery-soft as the first active test.
    if ctx in ("rough", "unknown") and proj_vx < args.min_rough_active_vx:
        return False, "rough_projection_too_slow_for_first_active"

    if ctx == "downslope" and proj_vx < args.min_downslope_active_vx:
        return False, "downslope_projection_too_slow_for_first_active"

    # First active candidate should be low-risk contexts only unless explicitly enabled later.
    if ctx not in args.allowed_contexts:
        return False, "context_not_allowed_for_first_active"

    return True, "accepted_dryrun"


ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out-csv", default="datasets/phase_j/j6_guarded_active_dryrun_v0.csv")
ap.add_argument("--out-md", default="reports/phase_j/j6_guarded_active_dryrun_summary_v0.md")
ap.add_argument("--max-abs-delta-vx", type=float, default=0.030)
ap.add_argument("--max-abs-delta-yaw", type=float, default=0.020)
ap.add_argument("--max-abs-delta-body-h", type=float, default=0.006)
ap.add_argument("--max-abs-delta-clearance", type=float, default=0.006)
ap.add_argument("--min-active-body-h", type=float, default=0.314)
ap.add_argument("--min-rough-active-vx", type=float, default=0.160)
ap.add_argument("--min-downslope-active-vx", type=float, default=0.150)
ap.add_argument("--allowed-contexts", nargs="+", default=["flat", "upslope", "downslope"])
args = ap.parse_args()

p = Path(args.csv)
if not p.exists():
    raise SystemExit(f"[TRACER] missing csv: {p}")

rows = list(csv.DictReader(open(p)))
if not rows:
    raise SystemExit(f"[TRACER] empty csv: {p}")

out_rows = []
reason_counts = Counter()
ctx_counts = Counter()
ctx_accept = defaultdict(lambda: {"n": 0, "accepted": 0})
ctx_action_accept = defaultdict(lambda: {"n": 0, "accepted": 0})

for r in rows:
    accepted, reason = decide(r, args)

    ctx = r.get("context", "unknown") or "unknown"
    action_id = r.get("action_id", "NA") or "NA"

    reason_counts[reason] += 1
    ctx_counts[ctx] += 1
    ctx_accept[ctx]["n"] += 1
    ctx_accept[ctx]["accepted"] += int(accepted)
    ctx_action_accept[(ctx, action_id)]["n"] += 1
    ctx_action_accept[(ctx, action_id)]["accepted"] += int(accepted)

    out = dict(r)
    out["j6_accept"] = "1" if accepted else "0"
    out["j6_reason"] = reason
    out_rows.append(out)

out_csv = Path(args.out_csv)
out_csv.parent.mkdir(parents=True, exist_ok=True)

fields = list(out_rows[0].keys())
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

out_md = Path(args.out_md)
out_md.parent.mkdir(parents=True, exist_ok=True)

accepted_total = sum(int(r["j6_accept"]) for r in out_rows)
accept_rate = accepted_total / len(out_rows)

cols = ["delta_vx", "delta_yaw_rate", "delta_body_h", "delta_clearance", "proj_vx", "proj_body_h", "proj_clearance"]
summary_all = {c: stats([ff(r.get(c)) for r in out_rows]) for c in cols}
summary_acc = {c: stats([ff(r.get(c)) for r in out_rows if r["j6_accept"] == "1"]) for c in cols}

with open(out_md, "w") as f:
    f.write("# TRACER Phase-J6 Guarded Active Dry-Run v0\n\n")
    f.write("This evaluates whether J4 projected references are safe candidates for a future guarded active test.\n\n")
    f.write(f"- input csv: `{p}`\n")
    f.write(f"- output csv: `{out_csv}`\n")
    f.write(f"- rows: `{len(out_rows)}`\n")
    f.write(f"- accepted rows: `{accepted_total}`\n")
    f.write(f"- accept rate: `{accept_rate:.6f}`\n")
    f.write(f"- allowed contexts: `{', '.join(args.allowed_contexts)}`\n\n")

    f.write("## Guard thresholds\n\n")
    f.write(f"- max_abs_delta_vx: `{args.max_abs_delta_vx}`\n")
    f.write(f"- max_abs_delta_yaw: `{args.max_abs_delta_yaw}`\n")
    f.write(f"- max_abs_delta_body_h: `{args.max_abs_delta_body_h}`\n")
    f.write(f"- max_abs_delta_clearance: `{args.max_abs_delta_clearance}`\n")
    f.write(f"- min_active_body_h: `{args.min_active_body_h}`\n")
    f.write(f"- min_rough_active_vx: `{args.min_rough_active_vx}`\n")
    f.write(f"- min_downslope_active_vx: `{args.min_downslope_active_vx}`\n\n")

    f.write("## Decision reasons\n\n")
    f.write("| reason | rows |\n")
    f.write("|---|---:|\n")
    for reason, cnt in reason_counts.most_common():
        f.write(f"| {reason} | {cnt} |\n")

    f.write("\n## Acceptance by context\n\n")
    f.write("| context | rows | accepted | accept rate |\n")
    f.write("|---|---:|---:|---:|\n")
    for ctx in sorted(ctx_accept):
        n = ctx_accept[ctx]["n"]
        a = ctx_accept[ctx]["accepted"]
        f.write(f"| {ctx} | {n} | {a} | {a / n if n else 0.0:.6f} |\n")

    f.write("\n## Acceptance by context/action\n\n")
    f.write("| context | action_id | rows | accepted | accept rate |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for (ctx, aid), d in sorted(ctx_action_accept.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = d["n"]
        a = d["accepted"]
        f.write(f"| {ctx} | {aid} | {n} | {a} | {a / n if n else 0.0:.6f} |\n")

    f.write("\n## Numeric summary: all vs accepted\n\n")
    f.write("| column | all mean | all min | all max | accepted mean | accepted min | accepted max |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for c in cols:
        sa = summary_all[c]
        sg = summary_acc[c]
        f.write(
            f"| {c} | {fmt(sa['mean'])} | {fmt(sa['min'])} | {fmt(sa['max'])} | "
            f"{fmt(sg['mean'])} | {fmt(sg['min'])} | {fmt(sg['max'])} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("- J6 is a dry-run gate only and does not modify active control.\n")
    f.write("- First active candidate should only use rows/actions that pass this dry-run gate.\n")
    f.write("- Goal/hold rows are protected and rejected by design.\n")
    f.write("- Rough/unknown aggressive recovery projections are rejected for the first active test unless a later explicit experiment enables them.\n")

print(f"[TRACER] wrote {out_csv}")
print(f"[TRACER] wrote {out_md}")
print(f"[TRACER] rows={len(out_rows)} accepted={accepted_total} accept_rate={accept_rate:.6f}")
