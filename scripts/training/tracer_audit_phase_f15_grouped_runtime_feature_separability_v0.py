#!/usr/bin/env python3
import csv
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev

IN_CSV = Path("datasets/phase_f/f13_d7_grouped_beta_calibration_dataset_v0.csv")
OUT_CSV = Path("datasets/phase_f/f15_grouped_runtime_feature_separability_v0.csv")
OUT_MD = Path("reports/phase_f/f15_grouped_runtime_feature_separability_summary_v0.md")

FEATURES = [
    "x", "y", "y_mean",
    "pred_beta_motion", "pred_beta_stability", "pred_beta_energy",
    "raw_beta_motion", "raw_beta_stability", "raw_beta_energy",
    "prior_beta_motion", "prior_beta_stability", "prior_beta_energy",
    "actual_beta_motion", "actual_beta_stability", "actual_beta_energy",
    "motion_score", "stability_score", "energy_score",
    "effort_proxy", "hold_drift",
    "rollout_max_abs_y", "rollout_mean_abs_y",
    "mean_abs_y_context", "max_abs_y_context",
    "ram_slip_proxy_mean", "ram_roughness_proxy_mean", "ram_sigma_mean",
    "ref_vx_mean", "ref_yaw_rate_mean", "ref_clearance_mean",
]

TARGETS = [
    "target_beta_motion_grouped_true_metric",
    "target_beta_stability_grouped_true_metric",
    "target_beta_energy_grouped_true_metric",
]

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

rows = list(csv.DictReader(open(IN_CSV)))
if not rows:
    raise SystemExit(f"empty input: {IN_CSV}")

groups = sorted(set(r.get("f13_base_tag", "unknown") for r in rows))

out_rows = []

for feat in FEATURES + TARGETS:
    vals_by_group = {}
    for g in groups:
        vals = [ff(r.get(feat)) for r in rows if r.get("f13_base_tag", "unknown") == g]
        vals = [v for v in vals if v is not None]
        if vals:
            vals_by_group[g] = vals

    if len(vals_by_group) < 2:
        continue

    means = {g: mean(vs) for g, vs in vals_by_group.items()}
    stds = {g: pstdev(vs) if len(vs) > 1 else 0.0 for g, vs in vals_by_group.items()}

    all_vals = [v for vs in vals_by_group.values() for v in vs]
    global_std = pstdev(all_vals) if len(all_vals) > 1 else 0.0
    between_range = max(means.values()) - min(means.values())

    # A rough separability score. Larger means group means differ relative to total variation.
    sep = between_range / max(global_std, 1e-12)

    row = {
        "feature": feat,
        "kind": "target" if feat in TARGETS else "runtime_feature",
        "global_std": global_std,
        "between_group_mean_range": between_range,
        "separability_range_over_std": sep,
    }

    for g in groups:
        row[f"{g}_mean"] = means.get(g, "")
        row[f"{g}_std"] = stds.get(g, "")
        row[f"{g}_n"] = len(vals_by_group.get(g, []))

    out_rows.append(row)

out_rows.sort(key=lambda r: float(r["separability_range_over_std"]), reverse=True)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fields = list(out_rows[0].keys())
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F15 Grouped Runtime Feature Separability Audit v0\n\n")
    f.write("This audits whether runtime-safe D7/log features separate the grouped true-metric beta target conditions: clean, m030, and p030.\n\n")
    f.write(f"- input: `{IN_CSV}`\n")
    f.write(f"- output csv: `{OUT_CSV}`\n")
    f.write(f"- rows: `{len(rows)}`\n")
    f.write(f"- groups: `{', '.join(groups)}`\n\n")

    f.write("## Top separable runtime features\n\n")
    f.write("| rank | feature | kind | range/std | clean mean | m030 mean | p030 mean |\n")
    f.write("|---:|---|---|---:|---:|---:|---:|\n")
    shown = 0
    for i, r in enumerate(out_rows, 1):
        if r["kind"] != "runtime_feature":
            continue
        f.write(
            f"| {shown+1} | {r['feature']} | {r['kind']} | "
            f"{float(r['separability_range_over_std']):.4f} | "
            f"{float(r.get('clean_mean', 0) or 0):.6f} | "
            f"{float(r.get('m030_mean', 0) or 0):.6f} | "
            f"{float(r.get('p030_mean', 0) or 0):.6f} |\n"
        )
        shown += 1
        if shown >= 15:
            break

    f.write("\n## Target separability\n\n")
    f.write("| target | range/std | clean mean | m030 mean | p030 mean |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for r in out_rows:
        if r["kind"] != "target":
            continue
        f.write(
            f"| {r['feature']} | "
            f"{float(r['separability_range_over_std']):.4f} | "
            f"{float(r.get('clean_mean', 0) or 0):.6f} | "
            f"{float(r.get('m030_mean', 0) or 0):.6f} | "
            f"{float(r.get('p030_mean', 0) or 0):.6f} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("If target beta separation is much larger than runtime-feature separation, the runtime-safe calibrator cannot reliably infer grouped true-metric beta targets from the current feature set. In that case, the next step is to add more informative runtime state/context features or collect more diverse terrain/objective rollouts.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print("[TRACER] top runtime separability:")
n = 0
for r in out_rows:
    if r["kind"] == "runtime_feature":
        print(f"  {r['feature']}: range/std={float(r['separability_range_over_std']):.4f}")
        n += 1
        if n >= 10:
            break
