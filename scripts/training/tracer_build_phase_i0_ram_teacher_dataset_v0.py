#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path
from collections import Counter
from statistics import mean, pstdev

IN_CSV = Path("datasets/phase_h/h0_objective_selector_training_table_v0.csv")
OUT_CSV = Path("datasets/phase_i/i0_ram_teacher_dataset_v0.csv")
OUT_JSON = Path("datasets/phase_i/i0_ram_teacher_manifest_v0.json")
OUT_MD = Path("reports/phase_i/i0_ram_teacher_dataset_summary_v0.md")

ID_COLS = ["tag", "base_tag", "n_timestep_rows"]

CANDIDATE_INPUT_FEATURES = [
    "context_frac_flat",
    "context_frac_start_flat",
    "context_frac_rough",
    "context_frac_upslope",
    "context_frac_downslope",
    "context_frac_goal_flat",
    "context_frac_unknown",
    "context_unique_count",
    "ref_vx_mean_mean_g1",
    "ref_vx_mean_std_g1",
    "ref_yaw_rate_mean_mean_g1",
    "ref_yaw_rate_mean_std_g1",
    "ref_clearance_mean_mean_g1",
    "ref_clearance_mean_std_g1",
    "x_progress_g1",
    "y_change_g1",
    "abs_y_change_g1",
    "y_late_minus_early_g1",
    "abs_y_late_minus_early_g1",
    "vx_tracking_error_g1",
    "vx_tracking_abs_error_g1",
    "vx_tracking_ratio_g1",
    "lateral_per_forward_g1",
]

SLIP_SIGNALS = [
    "vx_tracking_abs_error_g1",
    "lateral_per_forward_g1",
    "abs_y_change_g1",
    "abs_y_late_minus_early_g1",
]

ROUGH_SIGNALS = [
    "power_per_vx_g1",
    "contact_per_vx_g1",
    "imu_per_vx_g1",
]

UNCERTAINTY_SIGNALS = [
    "vx_tracking_abs_error_g1",
    "lateral_per_forward_g1",
    "imu_per_vx_g1",
]

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

def robust_minmax(rows, col):
    vals = [ff(r.get(col), None) for r in rows]
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None, None

    lo = vals[0]
    hi = vals[-1]
    if len(vals) >= 5:
        lo = vals[int(0.05 * (len(vals) - 1))]
        hi = vals[int(0.95 * (len(vals) - 1))]
    if abs(hi - lo) < 1e-12:
        hi = lo + 1.0
    return lo, hi

def norm_value(x, lo, hi):
    if x is None or lo is None or hi is None:
        return None
    return max(0.0, min(1.0, (float(x) - lo) / (hi - lo)))

def safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0

def safe_std(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) <= 1:
        return 0.0
    m = safe_mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

rows = list(csv.DictReader(open(IN_CSV)))
if not rows:
    raise SystemExit(f"[TRACER] empty input {IN_CSV}")

base_counts = Counter(r.get("base_tag", "") for r in rows)

all_signals = sorted(set(SLIP_SIGNALS + ROUGH_SIGNALS + UNCERTAINTY_SIGNALS))
scales = {c: robust_minmax(rows, c) for c in all_signals}

input_features = [c for c in CANDIDATE_INPUT_FEATURES if c in rows[0]]

out_rows = []
for r in rows:
    base = r.get("base_tag", "")
    n_base = base_counts.get(base, 1)

    slip_parts = []
    for c in SLIP_SIGNALS:
        lo, hi = scales.get(c, (None, None))
        slip_parts.append(norm_value(ff(r.get(c), None), lo, hi))
    rho_slip = safe_mean(slip_parts)

    rough_parts = []
    for c in ROUGH_SIGNALS:
        lo, hi = scales.get(c, (None, None))
        rough_parts.append(norm_value(ff(r.get(c), None), lo, hi))
    rho_rough = safe_mean(rough_parts)

    unc_parts = []
    for c in UNCERTAINTY_SIGNALS:
        lo, hi = scales.get(c, (None, None))
        unc_parts.append(norm_value(ff(r.get(c), None), lo, hi))

    coverage_unc = 1.0 / math.sqrt(max(float(n_base), 1.0))
    # keep p045/new sparse anchors visibly uncertain
    sigma = 0.55 * safe_mean(unc_parts) + 0.45 * coverage_unc
    sigma = max(0.0, min(1.0, sigma))

    out = {
        "tag": r.get("tag", ""),
        "base_tag": base,
        "n_timestep_rows": r.get("n_timestep_rows", ""),
        "base_condition_count": str(n_base),
        "rho_slip_target": f"{rho_slip:.8f}",
        "rho_rough_target": f"{rho_rough:.8f}",
        "sigma_target": f"{sigma:.8f}",
        "is_sparse_anchor": "1" if n_base < 3 else "0",
    }

    for c in input_features:
        out[c] = r.get(c, "")

    out_rows.append(out)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "tag",
    "base_tag",
    "n_timestep_rows",
    "base_condition_count",
    "rho_slip_target",
    "rho_rough_target",
    "sigma_target",
    "is_sparse_anchor",
] + input_features

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

manifest = {
    "phase": "I0",
    "purpose": "RAM supervised pretraining teacher dataset",
    "input_csv": str(IN_CSV),
    "output_csv": str(OUT_CSV),
    "n_rows": len(out_rows),
    "input_features": input_features,
    "target_cols": ["rho_slip_target", "rho_rough_target", "sigma_target"],
    "teacher_construction": {
        "rho_slip_target": SLIP_SIGNALS,
        "rho_rough_target": ROUGH_SIGNALS,
        "sigma_target": {
            "signals": UNCERTAINTY_SIGNALS,
            "coverage_uncertainty": "1/sqrt(base_condition_count)",
            "formula": "0.55*mean(normalized uncertainty signals)+0.45*coverage_uncertainty",
        },
    },
    "scales": {k: {"lo": v[0], "hi": v[1]} for k, v in scales.items()},
    "safe_claim": "Heuristic RAM teacher seeds for supervised pretraining; not final teacher-student RAM.",
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(manifest, indent=2))

def col_vals(col):
    return [ff(r[col], None) for r in out_rows if ff(r.get(col), None) is not None]

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-I0 RAM Teacher Dataset v0\n\n")
    f.write("This builds heuristic RAM teacher targets for supervised pretraining.\n\n")
    f.write(f"- input: `{IN_CSV}`\n")
    f.write(f"- output csv: `{OUT_CSV}`\n")
    f.write(f"- manifest: `{OUT_JSON}`\n")
    f.write(f"- rows: `{len(out_rows)}`\n")
    f.write(f"- input features: `{len(input_features)}`\n\n")

    f.write("## Rows by base condition\n\n")
    f.write("| base_tag | rows |\n")
    f.write("|---|---:|\n")
    for k in sorted(base_counts):
        f.write(f"| {k} | {base_counts[k]} |\n")

    f.write("\n## Target summary\n\n")
    f.write("| target | mean | std | min | max |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for c in ["rho_slip_target", "rho_rough_target", "sigma_target"]:
        vals = col_vals(c)
        f.write(
            f"| {c} | {safe_mean(vals):.6f} | {safe_std(vals):.6f} | "
            f"{min(vals):.6f} | {max(vals):.6f} |\n"
        )

    f.write("\n## Per-condition target means\n\n")
    f.write("| base_tag | n | rho_slip | rho_rough | sigma |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for k in sorted(base_counts):
        rs = [r for r in out_rows if r["base_tag"] == k]
        f.write(
            f"| {k} | {len(rs)} | "
            f"{safe_mean(col_vals_for := [ff(r['rho_slip_target']) for r in rs]):.6f} | "
            f"{safe_mean([ff(r['rho_rough_target']) for r in rs]):.6f} | "
            f"{safe_mean([ff(r['sigma_target']) for r in rs]):.6f} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("These are heuristic RAM teacher seeds derived from tracking/lateral/contact/energy proxies. They are useful for RAM supervised pretraining and runtime plumbing, but they are not yet a final teacher-student RAM learned from real proprioceptive prediction error.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_JSON}")
print(f"[TRACER] wrote {OUT_MD}")
print(f"[TRACER] rows={len(out_rows)} input_features={len(input_features)}")
