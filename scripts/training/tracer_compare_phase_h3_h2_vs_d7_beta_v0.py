#!/usr/bin/env python3
import argparse
import bisect
import csv
import math
from pathlib import Path
from collections import defaultdict
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


def fmt(v):
    return "NA" if v is None else f"{float(v):.6f}"


def fmt_beta(v):
    if v is None:
        return "NA"
    return "(" + ", ".join(f"{float(x):.4f}" for x in v) + ")"


def stats(vals):
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def read_csv(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[TRACER] missing csv: {p}")
    rows = list(csv.DictReader(open(p)))
    if not rows:
        raise SystemExit(f"[TRACER] empty csv: {p}")
    return rows


def find_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def detect_h2_beta_cols(cols):
    trio = [
        find_col(cols, ["beta_motion", "h2_beta_motion"]),
        find_col(cols, ["beta_stability", "h2_beta_stability"]),
        find_col(cols, ["beta_energy", "h2_beta_energy"]),
    ]
    if not all(trio):
        raise SystemExit(f"[TRACER] cannot detect H2 beta columns. columns={list(cols)}")
    return trio


def detect_d7_beta_groups(cols):
    candidates = {
        "published_beta": [
            ["beta_motion", "objective_beta_motion", "calibrated_beta_motion", "out_beta_motion"],
            ["beta_stability", "objective_beta_stability", "calibrated_beta_stability", "out_beta_stability"],
            ["beta_energy", "objective_beta_energy", "calibrated_beta_energy", "out_beta_energy"],
        ],
        "actual_beta": [
            ["actual_beta_motion", "actual_motion", "beta_actual_motion"],
            ["actual_beta_stability", "actual_stability", "beta_actual_stability"],
            ["actual_beta_energy", "actual_energy", "beta_actual_energy"],
        ],
        "pred_beta": [
            ["pred_beta_motion", "pred_motion", "model_beta_motion"],
            ["pred_beta_stability", "pred_stability", "model_beta_stability"],
            ["pred_beta_energy", "pred_energy", "model_beta_energy"],
        ],
        "prior_beta": [
            ["prior_beta_motion", "prior_motion"],
            ["prior_beta_stability", "prior_stability"],
            ["prior_beta_energy", "prior_energy"],
        ],
        "raw_beta": [
            ["raw_beta_motion", "raw_motion"],
            ["raw_beta_stability", "raw_stability"],
            ["raw_beta_energy", "raw_energy"],
        ],
        "static_beta": [
            ["static_beta_motion"],
            ["static_beta_stability"],
            ["static_beta_energy"],
        ],
    }

    groups = {}
    for name, cand_trios in candidates.items():
        trio = [find_col(cols, cands) for cands in cand_trios]
        if all(trio):
            groups[name] = trio

    if not groups:
        raise SystemExit(
            "[TRACER] cannot detect any D7 beta group.\n"
            f"columns={list(cols)}\n"
            "Run: head -1 <D7_CSV>"
        )

    return groups


def get_time_col(cols):
    c = find_col(cols, ["t_wall", "time", "timestamp", "stamp", "t"])
    if c is None:
        raise SystemExit(f"[TRACER] cannot detect time column. columns={list(cols)}")
    return c


def get_context_col(cols):
    return find_col(cols, ["context", "terrain_context", "terrain_context_label", "label"])


def beta_from_row(row, cols):
    vals = [ff(row.get(c), None) for c in cols]
    if any(v is None for v in vals):
        return None
    return vals


def beta_l1(a, b):
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b))


def nearest_by_time(t, times, rows):
    i = bisect.bisect_left(times, t)
    best = None
    for j in [i - 1, i, i + 1]:
        if 0 <= j < len(times):
            dt = abs(times[j] - t)
            if best is None or dt < best[0]:
                best = (dt, rows[j])
    return best


def aggregate(records, key_name, key_fn):
    out = []
    buckets = defaultdict(list)
    for r in records:
        buckets[key_fn(r)].append(r)

    for k in sorted(buckets):
        rs = buckets[k]
        l1 = stats([ff(r["l1"]) for r in rs])
        dt = stats([ff(r["dt_abs"]) for r in rs])

        h2_mean = []
        d7_mean = []
        for c in ["motion", "stability", "energy"]:
            h2_mean.append(stats([ff(r[f"h2_beta_{c}"]) for r in rs])["mean"])
            d7_mean.append(stats([ff(r[f"d7_beta_{c}"]) for r in rs])["mean"])

        out.append({
            key_name: k,
            "n": len(rs),
            "mean_l1": l1["mean"],
            "std_l1": l1["std"],
            "max_l1": l1["max"],
            "mean_dt_abs": dt["mean"],
            "h2_mean": h2_mean,
            "d7_mean": d7_mean,
        })
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--h2-csv", required=True)
ap.add_argument("--d7-csv", required=True)
ap.add_argument("--out-csv", default="datasets/phase_h/h3_h2_vs_d7_beta_comparison_v0.csv")
ap.add_argument("--out-md", default="reports/phase_h/h3_h2_vs_d7_beta_comparison_summary_v0.md")
ap.add_argument("--max-dt", type=float, default=0.35)
args = ap.parse_args()

h2_rows = read_csv(args.h2_csv)
d7_rows = read_csv(args.d7_csv)

h2_cols = h2_rows[0].keys()
d7_cols = d7_rows[0].keys()

h2_t_col = get_time_col(h2_cols)
d7_t_col = get_time_col(d7_cols)

h2_ctx_col = get_context_col(h2_cols)
d7_ctx_col = get_context_col(d7_cols)

h2_beta_cols = detect_h2_beta_cols(h2_cols)
d7_groups = detect_d7_beta_groups(d7_cols)

d7_time_rows = []
for r in d7_rows:
    t = ff(r.get(d7_t_col), None)
    if t is not None:
        d7_time_rows.append((t, r))

d7_time_rows.sort(key=lambda x: x[0])
d7_times = [x[0] for x in d7_time_rows]
d7_sorted_rows = [x[1] for x in d7_time_rows]

records = []
skipped = 0

for h2 in h2_rows:
    t = ff(h2.get(h2_t_col), None)
    h2_beta = beta_from_row(h2, h2_beta_cols)
    if t is None or h2_beta is None:
        skipped += 1
        continue

    nearest = nearest_by_time(t, d7_times, d7_sorted_rows)
    if nearest is None:
        skipped += 1
        continue

    dt_abs, d7 = nearest
    if dt_abs > args.max_dt:
        skipped += 1
        continue

    h2_context = h2.get(h2_ctx_col, "unknown") if h2_ctx_col else "unknown"
    d7_context = d7.get(d7_ctx_col, "") if d7_ctx_col else ""

    for group_name, d7_beta_cols in d7_groups.items():
        d7_beta = beta_from_row(d7, d7_beta_cols)
        if d7_beta is None:
            continue

        l1 = beta_l1(h2_beta, d7_beta)

        records.append({
            "d7_group": group_name,
            "t_h2": f"{t:.6f}",
            "t_d7": f"{ff(d7.get(d7_t_col), 0.0):.6f}",
            "dt_abs": f"{dt_abs:.6f}",
            "h2_context": h2_context,
            "d7_context": d7_context,
            "known_context": "1" if h2_context != "unknown" else "0",
            "h2_beta_motion": f"{h2_beta[0]:.8f}",
            "h2_beta_stability": f"{h2_beta[1]:.8f}",
            "h2_beta_energy": f"{h2_beta[2]:.8f}",
            "d7_beta_motion": f"{d7_beta[0]:.8f}",
            "d7_beta_stability": f"{d7_beta[1]:.8f}",
            "d7_beta_energy": f"{d7_beta[2]:.8f}",
            "l1": f"{l1:.8f}",
        })

if not records:
    raise SystemExit(
        "[TRACER] no matched H2-D7 beta records. "
        f"h2_rows={len(h2_rows)} d7_rows={len(d7_rows)} skipped={skipped} max_dt={args.max_dt}"
    )

out_csv = Path(args.out_csv)
out_csv.parent.mkdir(parents=True, exist_ok=True)

fields = [
    "d7_group",
    "t_h2",
    "t_d7",
    "dt_abs",
    "h2_context",
    "d7_context",
    "known_context",
    "h2_beta_motion",
    "h2_beta_stability",
    "h2_beta_energy",
    "d7_beta_motion",
    "d7_beta_stability",
    "d7_beta_energy",
    "l1",
]

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(records)

out_md = Path(args.out_md)
out_md.parent.mkdir(parents=True, exist_ok=True)

groups = sorted(set(r["d7_group"] for r in records))

with open(out_md, "w") as f:
    f.write("# TRACER Phase-H3 H2 vs D7 Beta Comparison v0\n\n")
    f.write("This compares the Phase-H2 Objective Selector beta shadow against beta outputs logged by the existing D7 objective selector on the same rollout.\n\n")
    f.write(f"- H2 csv: `{args.h2_csv}`\n")
    f.write(f"- D7 csv: `{args.d7_csv}`\n")
    f.write(f"- output csv: `{out_csv}`\n")
    f.write(f"- matched records: `{len(records)}`\n")
    f.write(f"- skipped H2 rows: `{skipped}`\n")
    f.write(f"- max dt: `{args.max_dt}` seconds\n")
    f.write(f"- detected D7 beta groups: `{', '.join(groups)}`\n\n")

    f.write("## Overall comparison by D7 beta group\n\n")
    f.write("| D7 group | records | mean L1 | std L1 | max L1 | mean |H2-D7| dt | H2 beta mean | D7 beta mean |\n")
    f.write("|---|---:|---:|---:|---:|---:|---|---|\n")
    for row in aggregate(records, "d7_group", lambda r: r["d7_group"]):
        f.write(
            f"| {row['d7_group']} | {row['n']} | {fmt(row['mean_l1'])} | {fmt(row['std_l1'])} | "
            f"{fmt(row['max_l1'])} | {fmt(row['mean_dt_abs'])} | "
            f"{fmt_beta(row['h2_mean'])} | {fmt_beta(row['d7_mean'])} |\n"
        )

    f.write("\n## Known-context-only comparison by D7 beta group\n\n")
    known_records = [r for r in records if r["known_context"] == "1"]
    f.write("| D7 group | records | mean L1 | std L1 | max L1 | mean dt | H2 beta mean | D7 beta mean |\n")
    f.write("|---|---:|---:|---:|---:|---:|---|---|\n")
    for row in aggregate(known_records, "d7_group", lambda r: r["d7_group"]):
        f.write(
            f"| {row['d7_group']} | {row['n']} | {fmt(row['mean_l1'])} | {fmt(row['std_l1'])} | "
            f"{fmt(row['max_l1'])} | {fmt(row['mean_dt_abs'])} | "
            f"{fmt_beta(row['h2_mean'])} | {fmt_beta(row['d7_mean'])} |\n"
        )

    f.write("\n## Known-context-only comparison by terrain context\n\n")
    f.write("| D7 group | context | records | mean L1 | max L1 | H2 beta mean | D7 beta mean |\n")
    f.write("|---|---|---:|---:|---:|---|---|\n")
    for group in groups:
        group_records = [r for r in known_records if r["d7_group"] == group]
        for row in aggregate(group_records, "context", lambda r: r["h2_context"]):
            f.write(
                f"| {group} | {row['context']} | {row['n']} | {fmt(row['mean_l1'])} | "
                f"{fmt(row['max_l1'])} | {fmt_beta(row['h2_mean'])} | {fmt_beta(row['d7_mean'])} |\n"
            )

    f.write("\n## Safe interpretation\n\n")
    f.write("- This is a shadow comparison only. H2 beta is not used for active control here.\n")
    f.write("- Large H2-D7 L1 does not automatically mean H2 is wrong, because H2 is trained toward robust true-metric teacher seeds while D7 is the earlier runtime-aligned selector.\n")
    f.write("- If H2 is much more motion-heavy or frequently saturates at a simplex corner, keep it shadow-only and improve RAM/context coverage before active deployment.\n")
    f.write("- If terrain transitions are visible and L1 is moderate, H2 can remain the candidate Objective Selector pretraining model for the next RAM/RL stages.\n")

print(f"[TRACER] wrote {out_csv}")
print(f"[TRACER] wrote {out_md}")
print(f"[TRACER] matched_records={len(records)} skipped={skipped} groups={groups}")
