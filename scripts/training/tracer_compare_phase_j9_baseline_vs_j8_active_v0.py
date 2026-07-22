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
    return "NA" if v is None else f"{float(v):.6f}"


def manifest_dirs(manifest_path):
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"empty manifest: {manifest_path}")
    r = rows[0]
    return Path(r["log_dir"]), Path(r["d5_log_dir"])


def read_csv(path):
    if not Path(path).exists():
        return []
    return list(csv.DictReader(open(path)))


def state_summary(policy_rows, goal_x=8.0):
    valid = [r for r in policy_rows if ff(r.get("x")) is not None and ff(r.get("y")) is not None]
    xs = [ff(r["x"]) for r in valid]
    ys = [ff(r["y"]) for r in valid]
    ts = [ff(r.get("t_wall")) for r in valid]
    ctx = Counter(r.get("context", "unknown") or "unknown" for r in valid)

    first_goal_t = None
    first_t = ts[0] if ts else None
    for r in valid:
        x = ff(r.get("x"))
        t = ff(r.get("t_wall"))
        if x is not None and t is not None and x >= goal_x:
            first_goal_t = t
            break

    return {
        "rows": len(policy_rows),
        "valid_xy": len(valid),
        "first_x": xs[0] if xs else None,
        "first_y": ys[0] if ys else None,
        "final_x": xs[-1] if xs else None,
        "final_y": ys[-1] if ys else None,
        "min_x": min(xs) if xs else None,
        "max_x": max(xs) if xs else None,
        "mean_abs_y": mean(abs(y) for y in ys) if ys else None,
        "max_abs_y": max(abs(y) for y in ys) if ys else None,
        "goal_reached": bool(xs and max(xs) >= goal_x),
        "duration_s": (ts[-1] - ts[0]) if len(ts) >= 2 and ts[0] is not None and ts[-1] is not None else None,
        "time_to_goal_s": (first_goal_t - first_t) if first_goal_t is not None and first_t is not None else None,
        "context_counts": ctx,
    }


def context_state_summary(policy_rows):
    by_ctx = defaultdict(list)
    for r in policy_rows:
        by_ctx[r.get("context", "unknown") or "unknown"].append(r)

    out = {}
    for ctx, rows in by_ctx.items():
        xs = [ff(r.get("x")) for r in rows]
        ys = [ff(r.get("y")) for r in rows]
        ref_vx = [ff(r.get("ref_vx")) for r in rows]
        ref_clearance = [ff(r.get("ref_clearance")) for r in rows]
        xs = [x for x in xs if x is not None]
        ys = [y for y in ys if y is not None]
        out[ctx] = {
            "rows": len(rows),
            "x_min": min(xs) if xs else None,
            "x_max": max(xs) if xs else None,
            "mean_abs_y": mean(abs(y) for y in ys) if ys else None,
            "max_abs_y": max(abs(y) for y in ys) if ys else None,
            "ref_vx_mean": stats(ref_vx)["mean"],
            "ref_clearance_mean": stats(ref_clearance)["mean"],
        }
    return out


def baseline_cmd_summary(gate_rows):
    by_ctx = defaultdict(list)
    source = Counter()
    for r in gate_rows:
        by_ctx[r.get("context", "unknown") or "unknown"].append(r)
        source[r.get("selected_source", "unknown") or "unknown"] += 1

    out = {}
    for ctx, rows in by_ctx.items():
        out[ctx] = {
            "rows": len(rows),
            "cmd_vx_mean": stats([ff(r.get("selected_vx")) for r in rows])["mean"],
            "cmd_clearance_mean": stats([ff(r.get("selected_clearance")) for r in rows])["mean"],
            "accept_rate": mean(1.0 if (r.get("gate_accept") == "True" or r.get("gate_accept") == "1") else 0.0 for r in rows) if rows else None,
        }
    return out, source


def active_cmd_summary(j7_rows):
    by_ctx = defaultdict(list)
    source = Counter()
    reason = Counter()
    for r in j7_rows:
        by_ctx[r.get("context", "unknown") or "unknown"].append(r)
        source[r.get("j7_source", "unknown") or "unknown"] += 1
        reason[r.get("j7_reason", "unknown") or "unknown"] += 1

    out = {}
    for ctx, rows in by_ctx.items():
        out[ctx] = {
            "rows": len(rows),
            "cmd_vx_mean": stats([ff(r.get("out_vx")) for r in rows])["mean"],
            "cmd_clearance_mean": stats([ff(r.get("out_clearance")) for r in rows])["mean"],
            "accept_rate": mean(1.0 if r.get("j7_accept") == "1" else 0.0 for r in rows) if rows else None,
            "projected_rate": mean(1.0 if r.get("j7_source") == "projected" else 0.0 for r in rows) if rows else None,
            "failsafe_rate": mean(1.0 if r.get("j7_source") == "failsafe_hold" else 0.0 for r in rows) if rows else None,
        }
    return out, source, reason


ap = argparse.ArgumentParser()
ap.add_argument("--baseline-manifest", required=True)
ap.add_argument("--active-manifest", required=True)
ap.add_argument("--active-j7-csv", required=True)
ap.add_argument("--goal-x", type=float, default=8.0)
ap.add_argument("--out-csv", default="datasets/phase_j/j9_baseline_vs_j8_active_comparison_v0.csv")
ap.add_argument("--out-md", default="reports/phase_j/j9_baseline_vs_j8_active_comparison_summary_v0.md")
args = ap.parse_args()

baseline_manifest = Path(args.baseline_manifest)
active_manifest = Path(args.active_manifest)
active_j7_csv = Path(args.active_j7_csv)

_, base_d5 = manifest_dirs(baseline_manifest)
_, active_d5 = manifest_dirs(active_manifest)

base_policy = read_csv(base_d5 / "policy_inputs_v0.csv")
active_policy = read_csv(active_d5 / "policy_inputs_v0.csv")
base_gate = read_csv(base_d5 / "gated_selector_dryrun_v0.csv")
active_j7 = read_csv(active_j7_csv)

base_state = state_summary(base_policy, args.goal_x)
active_state = state_summary(active_policy, args.goal_x)

base_ctx_state = context_state_summary(base_policy)
active_ctx_state = context_state_summary(active_policy)

base_cmd, base_source = baseline_cmd_summary(base_gate)
active_cmd, active_source, active_reason = active_cmd_summary(active_j7)

all_ctx = sorted(set(base_ctx_state) | set(active_ctx_state) | set(base_cmd) | set(active_cmd))

out_csv = Path(args.out_csv)
out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="") as f:
    fields = [
        "context",
        "baseline_rows", "active_rows",
        "baseline_mean_abs_y", "active_mean_abs_y", "delta_mean_abs_y",
        "baseline_max_abs_y", "active_max_abs_y", "delta_max_abs_y",
        "baseline_ref_vx_mean", "active_ref_vx_mean", "baseline_cmd_vx_mean", "active_cmd_vx_mean",
        "baseline_cmd_clearance_mean", "active_cmd_clearance_mean",
        "baseline_gate_accept_rate", "active_j7_accept_rate", "active_projected_rate", "active_failsafe_rate",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for ctx in all_ctx:
        bs = base_ctx_state.get(ctx, {})
        ac = active_ctx_state.get(ctx, {})
        bc = base_cmd.get(ctx, {})
        jc = active_cmd.get(ctx, {})
        bmean = bs.get("mean_abs_y")
        amean = ac.get("mean_abs_y")
        bmax = bs.get("max_abs_y")
        amax = ac.get("max_abs_y")
        w.writerow({
            "context": ctx,
            "baseline_rows": bs.get("rows", 0),
            "active_rows": ac.get("rows", 0),
            "baseline_mean_abs_y": bmean,
            "active_mean_abs_y": amean,
            "delta_mean_abs_y": None if bmean is None or amean is None else amean - bmean,
            "baseline_max_abs_y": bmax,
            "active_max_abs_y": amax,
            "delta_max_abs_y": None if bmax is None or amax is None else amax - bmax,
            "baseline_ref_vx_mean": bs.get("ref_vx_mean"),
            "active_ref_vx_mean": ac.get("ref_vx_mean"),
            "baseline_cmd_vx_mean": bc.get("cmd_vx_mean"),
            "active_cmd_vx_mean": jc.get("cmd_vx_mean"),
            "baseline_cmd_clearance_mean": bc.get("cmd_clearance_mean"),
            "active_cmd_clearance_mean": jc.get("cmd_clearance_mean"),
            "baseline_gate_accept_rate": bc.get("accept_rate"),
            "active_j7_accept_rate": jc.get("accept_rate"),
            "active_projected_rate": jc.get("projected_rate"),
            "active_failsafe_rate": jc.get("failsafe_rate"),
        })

out_md = Path(args.out_md)
out_md.parent.mkdir(parents=True, exist_ok=True)

with open(out_md, "w") as f:
    f.write("# TRACER Phase-J9 Baseline vs J8 Guarded Active Comparison v0\n\n")
    f.write("This compares the standard D7/D6 gated baseline against the first J8 guarded-active meta-action smoke test.\n\n")
    f.write(f"- baseline manifest: `{baseline_manifest}`\n")
    f.write(f"- active manifest: `{active_manifest}`\n")
    f.write(f"- active J7 csv: `{active_j7_csv}`\n")
    f.write(f"- output csv: `{out_csv}`\n\n")

    f.write("## Mission-level outcome\n\n")
    f.write("| metric | baseline | J8 guarded active | delta active-baseline |\n")
    f.write("|---|---:|---:|---:|\n")
    for k in ["final_x", "final_y", "max_x", "mean_abs_y", "max_abs_y", "duration_s", "time_to_goal_s"]:
        b = base_state[k]
        a = active_state[k]
        d = None if b is None or a is None else a - b
        f.write(f"| {k} | {fmt(b)} | {fmt(a)} | {fmt(d)} |\n")
    f.write(f"| goal_reached | {base_state['goal_reached']} | {active_state['goal_reached']} | NA |\n")

    f.write("\n## Context counts\n\n")
    f.write("| context | baseline rows | active rows |\n")
    f.write("|---|---:|---:|\n")
    for ctx in all_ctx:
        f.write(f"| {ctx} | {base_ctx_state.get(ctx, {}).get('rows', 0)} | {active_ctx_state.get(ctx, {}).get('rows', 0)} |\n")

    f.write("\n## Segment-level lateral metrics\n\n")
    f.write("| context | baseline mean | active mean | delta mean | baseline max | active max | delta max |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for ctx in all_ctx:
        bs = base_ctx_state.get(ctx, {})
        ac = active_ctx_state.get(ctx, {})
        bmean = bs.get("mean_abs_y")
        amean = ac.get("mean_abs_y")
        bmax = bs.get("max_abs_y")
        amax = ac.get("max_abs_y")
        f.write(
            f"| {ctx} | {fmt(bmean)} | {fmt(amean)} | {fmt(None if bmean is None or amean is None else amean-bmean)} | "
            f"{fmt(bmax)} | {fmt(amax)} | {fmt(None if bmax is None or amax is None else amax-bmax)} |\n"
        )

    f.write("\n## Command-level comparison by context\n\n")
    f.write("| context | baseline cmd vx | active out vx | baseline clr | active out clr | base gate accept | J7 accept | projected rate | failsafe rate |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for ctx in all_ctx:
        bc = base_cmd.get(ctx, {})
        jc = active_cmd.get(ctx, {})
        f.write(
            f"| {ctx} | {fmt(bc.get('cmd_vx_mean'))} | {fmt(jc.get('cmd_vx_mean'))} | "
            f"{fmt(bc.get('cmd_clearance_mean'))} | {fmt(jc.get('cmd_clearance_mean'))} | "
            f"{fmt(bc.get('accept_rate'))} | {fmt(jc.get('accept_rate'))} | {fmt(jc.get('projected_rate'))} | {fmt(jc.get('failsafe_rate'))} |\n"
        )

    f.write("\n## Source / reason counts\n\n")
    f.write("### Baseline D5 selected source\n\n")
    f.write("| source | rows |\n|---|---:|\n")
    for k, v in base_source.most_common():
        f.write(f"| {k} | {v} |\n")

    f.write("\n### J8 J7 output source\n\n")
    f.write("| source | rows |\n|---|---:|\n")
    for k, v in active_source.most_common():
        f.write(f"| {k} | {v} |\n")

    f.write("\n### J8 J7 decision reason\n\n")
    f.write("| reason | rows |\n|---|---:|\n")
    for k, v in active_reason.most_common():
        f.write(f"| {k} | {v} |\n")

    f.write("\n## Safe interpretation\n\n")
    f.write("- J9 is not claiming global improvement yet; it checks whether J8 active intervention improves or preserves mission behavior.\n")
    f.write("- Flat/upslope are intervention segments where projected references were allowed.\n")
    f.write("- Rough/downslope/goal are protected segments where the desired result is non-degradation.\n")
    f.write("- If J8 reaches the goal with comparable or lower lateral drift, the next step is conservative rough/downslope opening rather than broad active control.\n")

print(f"[TRACER] wrote {out_csv}")
print(f"[TRACER] wrote {out_md}")
print(f"[TRACER] baseline goal={base_state['goal_reached']} active goal={active_state['goal_reached']}")
print(f"[TRACER] baseline max_abs_y={fmt(base_state['max_abs_y'])} active max_abs_y={fmt(active_state['max_abs_y'])}")
