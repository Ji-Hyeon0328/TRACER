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


def sat_rate(vals, eps=0.02):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    n = len(vals)
    s = sum(1 for v in vals if v <= eps or v >= 1.0 - eps)
    return s / n


def bucket(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[r.get(key, "unknown") or "unknown"].append(r)
    return d


ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out-csv", default="datasets/phase_i/i3_i2_vs_legacy_ram_by_context_v0.csv")
ap.add_argument("--out-md", default="reports/phase_i/i3_i2_vs_legacy_ram_by_context_summary_v0.md")
args = ap.parse_args()

p = Path(args.csv)
if not p.exists():
    raise SystemExit(f"[TRACER] missing csv: {p}")

rows = list(csv.DictReader(open(p)))
if not rows:
    raise SystemExit(f"[TRACER] empty csv: {p}")

pred_cols = ["rho_slip_pred", "rho_rough_pred", "sigma_pred"]
legacy_cols = ["legacy_slip", "legacy_rough", "legacy_sigma"]

def summarize(name, rs):
    out = {
        "group": name,
        "rows": len(rs),
    }

    valid_legacy = [
        r for r in rs
        if all(ff(r.get(c)) is not None for c in legacy_cols)
    ]
    out["legacy_valid_rows"] = len(valid_legacy)

    for c in pred_cols + legacy_cols + ["legacy_l1", "observed_features", "imputed_features", "x", "y", "ref_vx"]:
        st = stats([ff(r.get(c)) for r in rs])
        out[f"{c}_n"] = st["n"]
        out[f"{c}_mean"] = st["mean"]
        out[f"{c}_std"] = st["std"]
        out[f"{c}_min"] = st["min"]
        out[f"{c}_max"] = st["max"]

    for c in pred_cols:
        out[f"{c}_sat_rate"] = sat_rate([ff(r.get(c)) for r in rs])

    # Mean vector L1 manually, only where legacy exists.
    l1s = []
    for r in valid_legacy:
        pred = [ff(r.get(c)) for c in pred_cols]
        leg = [ff(r.get(c)) for c in legacy_cols]
        if all(v is not None for v in pred + leg):
            l1s.append(sum(abs(pred[i] - leg[i]) for i in range(3)))
    st_l1 = stats(l1s)
    out["manual_l1_mean"] = st_l1["mean"]
    out["manual_l1_std"] = st_l1["std"]
    out["manual_l1_max"] = st_l1["max"]

    return out


summaries = []
summaries.append(summarize("ALL", rows))
known = [r for r in rows if (r.get("context", "unknown") or "unknown") != "unknown"]
summaries.append(summarize("KNOWN_CONTEXT_ONLY", known))

for ctx, rs in sorted(bucket(rows, "context").items()):
    summaries.append(summarize(f"context:{ctx}", rs))

out_csv = Path(args.out_csv)
out_csv.parent.mkdir(parents=True, exist_ok=True)

fields = [
    "group", "rows", "legacy_valid_rows",
    "rho_slip_pred_mean", "rho_slip_pred_std", "rho_slip_pred_min", "rho_slip_pred_max", "rho_slip_pred_sat_rate",
    "rho_rough_pred_mean", "rho_rough_pred_std", "rho_rough_pred_min", "rho_rough_pred_max", "rho_rough_pred_sat_rate",
    "sigma_pred_mean", "sigma_pred_std", "sigma_pred_min", "sigma_pred_max", "sigma_pred_sat_rate",
    "legacy_slip_mean", "legacy_rough_mean", "legacy_sigma_mean",
    "legacy_l1_mean", "legacy_l1_std", "legacy_l1_max",
    "manual_l1_mean", "manual_l1_std", "manual_l1_max",
    "observed_features_mean", "imputed_features_mean",
    "x_mean", "y_mean", "ref_vx_mean",
]

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for s in summaries:
        w.writerow({k: s.get(k, "") for k in fields})

out_md = Path(args.out_md)
out_md.parent.mkdir(parents=True, exist_ok=True)

with open(out_md, "w") as f:
    f.write("# TRACER Phase-I3 I2 vs Legacy RAM Comparison v0\n\n")
    f.write("This compares Phase-I2 learned RAM shadow outputs against the legacy/proxy RAM stream by terrain context.\n\n")
    f.write(f"- input csv: `{p}`\n")
    f.write(f"- output csv: `{out_csv}`\n")
    f.write(f"- rows: `{len(rows)}`\n\n")

    f.write("## Overall / known-context summary\n\n")
    f.write("| group | rows | legacy rows | I2 slip | I2 rough | I2 sigma | legacy slip | legacy rough | legacy sigma | L1 vs legacy | sat slip | sat rough | sat sigma | obs/imputed |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for s in summaries[:2]:
        f.write(
            f"| {s['group']} | {s['rows']} | {s['legacy_valid_rows']} | "
            f"{fmt(s['rho_slip_pred_mean'])} | {fmt(s['rho_rough_pred_mean'])} | {fmt(s['sigma_pred_mean'])} | "
            f"{fmt(s['legacy_slip_mean'])} | {fmt(s['legacy_rough_mean'])} | {fmt(s['legacy_sigma_mean'])} | "
            f"{fmt(s['manual_l1_mean'])} | "
            f"{fmt(s['rho_slip_pred_sat_rate'])} | {fmt(s['rho_rough_pred_sat_rate'])} | {fmt(s['sigma_pred_sat_rate'])} | "
            f"{fmt(s['observed_features_mean'])}/{fmt(s['imputed_features_mean'])} |\n"
        )

    f.write("\n## By context\n\n")
    f.write("| context | rows | legacy rows | I2 slip | I2 rough | I2 sigma | legacy slip | legacy rough | legacy sigma | L1 vs legacy | sat slip | sat rough | sat sigma | y mean |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in summaries[2:]:
        ctx = s["group"].replace("context:", "")
        f.write(
            f"| {ctx} | {s['rows']} | {s['legacy_valid_rows']} | "
            f"{fmt(s['rho_slip_pred_mean'])} | {fmt(s['rho_rough_pred_mean'])} | {fmt(s['sigma_pred_mean'])} | "
            f"{fmt(s['legacy_slip_mean'])} | {fmt(s['legacy_rough_mean'])} | {fmt(s['legacy_sigma_mean'])} | "
            f"{fmt(s['manual_l1_mean'])} | "
            f"{fmt(s['rho_slip_pred_sat_rate'])} | {fmt(s['rho_rough_pred_sat_rate'])} | {fmt(s['sigma_pred_sat_rate'])} | "
            f"{fmt(s['y_mean'])} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("- I2 is a learned RAM shadow trained on heuristic teacher seeds, not a replacement for legacy RAM yet.\n")
    f.write("- Large L1 from legacy RAM is expected if I2 is capturing different stress/mismatch targets.\n")
    f.write("- High saturation rates indicate that the ridge+clamp model is too sharp for active deployment.\n")
    f.write("- The next step should either damp/calibrate I2 outputs or use them only as auxiliary shadow features for H2/H3 diagnostics.\n")

print(f"[TRACER] wrote {out_csv}")
print(f"[TRACER] wrote {out_md}")
print(f"[TRACER] rows={len(rows)} known_context_rows={len(known)}")
