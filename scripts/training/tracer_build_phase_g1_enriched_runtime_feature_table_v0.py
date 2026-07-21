#!/usr/bin/env python3
import csv
import math
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean, pstdev

IN_F19 = Path("datasets/phase_f/f19_d7_robust_beta_calibration_dataset_v0.csv")
OUT_CSV = Path("datasets/phase_g/g1_enriched_runtime_feature_table_v0.csv")
OUT_MD = Path("reports/phase_g/g1_enriched_runtime_feature_table_summary_v0.md")

TARGETS = [
    "target_beta_motion_robust_true_metric",
    "target_beta_stability_robust_true_metric",
    "target_beta_energy_robust_true_metric",
]

NUMERIC_COLS = [
    "x", "y", "reset_y", "y_mean",
    "pred_beta_motion", "pred_beta_stability", "pred_beta_energy",
    "raw_beta_motion", "raw_beta_stability", "raw_beta_energy",
    "prior_beta_motion", "prior_beta_stability", "prior_beta_energy",
    "actual_beta_motion", "actual_beta_stability", "actual_beta_energy",
    "err_beta_motion", "err_beta_stability", "err_beta_energy",
    "motion_score", "stability_score", "energy_score", "effort_proxy",
    "ram_slip_proxy_mean", "ram_roughness_proxy_mean", "ram_sigma_mean",
    "ref_vx_mean", "ref_yaw_rate_mean", "ref_clearance_mean",
    "rollout_true_metric_base_vx_mean",
    "rollout_true_metric_base_vy_abs_mean",
    "rollout_true_metric_joint_abs_power_mean",
    "rollout_true_metric_imu_ang_vel_norm_mean",
    "rollout_true_metric_contact_force_z_sum_mean",
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

def safe_mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return mean(vals) if vals else ""

def safe_std(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return pstdev(vals) if len(vals) > 1 else 0.0 if vals else ""

def safe_min(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return min(vals) if vals else ""

def safe_max(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return max(vals) if vals else ""

def first_valid(rows, col):
    for r in rows:
        v = r.get(col, "")
        if v != "":
            return v
    return ""

def num_first(rows, col, default=0.0):
    return ff(first_valid(rows, col), default)

def tercile_stats(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return "", "", "", ""
    n = len(vals)
    k = max(1, n // 3)
    early = vals[:k]
    mid = vals[n//3: 2*n//3] or vals
    late = vals[-k:]
    return safe_mean(early), safe_mean(mid), safe_mean(late), safe_mean(late) - safe_mean(early)

def context_fraction(rows):
    c = Counter(r.get("context", "unknown") for r in rows)
    n = len(rows) or 1
    out = {}
    for key in ["flat", "start_flat", "rough", "upslope", "downslope", "goal_flat", "unknown"]:
        out[f"context_frac_{key}"] = c.get(key, 0) / n
    out["context_mode"] = c.most_common(1)[0][0] if c else "unknown"
    out["context_unique_count"] = len(c)
    return out

def fmt(v):
    if v == "":
        return ""
    return f"{float(v):.6f}"

rows = list(csv.DictReader(open(IN_F19)))
if not rows:
    raise SystemExit(f"empty input: {IN_F19}")

groups = defaultdict(list)
for r in rows:
    tag = r.get("f19_source_tag", "unknown")
    groups[tag].append(r)

out_rows = []
for tag, rs in sorted(groups.items()):
    base_tag = first_valid(rs, "f19_base_tag") or tag

    out = {
        "tag": tag,
        "base_tag": base_tag,
        "n_timestep_rows": len(rs),
        "manifest": first_valid(rs, "f19_manifest"),
        "d7_csv": first_valid(rs, "f19_d7_csv"),
    }

    for t in TARGETS:
        out[t] = first_valid(rs, t)

    for c in NUMERIC_COLS:
        vals = [ff(r.get(c)) for r in rs]
        out[f"{c}_mean_g1"] = safe_mean(vals)
        out[f"{c}_std_g1"] = safe_std(vals)
        out[f"{c}_min_g1"] = safe_min(vals)
        out[f"{c}_max_g1"] = safe_max(vals)

    # Lateral temporal recovery features
    y_vals = [ff(r.get("y")) for r in rs]
    ay_vals = [abs(v) for v in y_vals if v is not None]
    y_early, y_mid, y_late, y_late_minus_early = tercile_stats(y_vals)
    ay_early, ay_mid, ay_late, ay_late_minus_early = tercile_stats(ay_vals)

    out["y_early_mean_g1"] = y_early
    out["y_mid_mean_g1"] = y_mid
    out["y_late_mean_g1"] = y_late
    out["y_late_minus_early_g1"] = y_late_minus_early
    out["abs_y_early_mean_g1"] = ay_early
    out["abs_y_mid_mean_g1"] = ay_mid
    out["abs_y_late_mean_g1"] = ay_late
    out["abs_y_late_minus_early_g1"] = ay_late_minus_early

    # Progress and tracking-like proxies.
    x0 = ff(rs[0].get("x"), 0.0)
    x1 = ff(rs[-1].get("x"), x0)
    y0 = ff(rs[0].get("y"), 0.0)
    y1 = ff(rs[-1].get("y"), y0)
    out["x_progress_g1"] = x1 - x0
    out["y_change_g1"] = y1 - y0
    out["abs_y_change_g1"] = abs(y1) - abs(y0)

    ref_vx = ff(out.get("ref_vx_mean_mean_g1"), None)
    true_vx = num_first(rs, "rollout_true_metric_base_vx_mean", None)
    true_vy_abs = num_first(rs, "rollout_true_metric_base_vy_abs_mean", None)
    power = num_first(rs, "rollout_true_metric_joint_abs_power_mean", None)
    imu = num_first(rs, "rollout_true_metric_imu_ang_vel_norm_mean", None)
    contact = num_first(rs, "rollout_true_metric_contact_force_z_sum_mean", None)

    eps = 1e-6
    if ref_vx is not None and true_vx is not None:
        out["vx_tracking_error_g1"] = ref_vx - true_vx
        out["vx_tracking_abs_error_g1"] = abs(ref_vx - true_vx)
        out["vx_tracking_ratio_g1"] = true_vx / max(abs(ref_vx), eps)
    else:
        out["vx_tracking_error_g1"] = ""
        out["vx_tracking_abs_error_g1"] = ""
        out["vx_tracking_ratio_g1"] = ""

    if true_vx is not None and power is not None:
        out["power_per_vx_g1"] = power / max(abs(true_vx), eps)
    else:
        out["power_per_vx_g1"] = ""

    if true_vx is not None and contact is not None:
        out["contact_per_vx_g1"] = contact / max(abs(true_vx), eps)
    else:
        out["contact_per_vx_g1"] = ""

    if true_vx is not None and true_vy_abs is not None:
        out["lateral_per_forward_g1"] = true_vy_abs / max(abs(true_vx), eps)
    else:
        out["lateral_per_forward_g1"] = ""

    if imu is not None and true_vx is not None:
        out["imu_per_vx_g1"] = imu / max(abs(true_vx), eps)
    else:
        out["imu_per_vx_g1"] = ""

    # Beta dynamics over rollout
    for prefix in ["pred_beta", "raw_beta", "prior_beta", "actual_beta"]:
        for comp in ["motion", "stability", "energy"]:
            col = f"{prefix}_{comp}"
            vals = [ff(r.get(col)) for r in rs]
            e, m, l, d = tercile_stats(vals)
            out[f"{col}_early_mean_g1"] = e
            out[f"{col}_mid_mean_g1"] = m
            out[f"{col}_late_mean_g1"] = l
            out[f"{col}_late_minus_early_g1"] = d

    out.update(context_fraction(rs))
    out_rows.append(out)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fields = list(out_rows[0].keys())
for r in out_rows:
    for k in r:
        if k not in fields:
            fields.append(k)

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

# Simple feature-target absolute correlation ranking.
feature_cols = [
    c for c in fields
    if c not in ["tag", "base_tag", "manifest", "d7_csv", "context_mode"]
    and not c.startswith("target_beta_")
]

def corr(xs, ys):
    xs = [ff(x) for x in xs]
    ys = [ff(y) for y in ys]
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 4:
        return None
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mx, my = mean(x), mean(y)
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)

corr_rows = []
for fc in feature_cols:
    vals = [r.get(fc, "") for r in out_rows]
    for tc in TARGETS:
        c = corr(vals, [r.get(tc, "") for r in out_rows])
        if c is not None:
            corr_rows.append((abs(c), c, fc, tc))

corr_rows.sort(reverse=True)

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-G1 Enriched Runtime Feature Table v0\n\n")
    f.write("This aggregates F19 D7 timestep logs into rollout-level enriched RAM/context features for true-metric beta prediction diagnostics.\n\n")
    f.write(f"- input: `{IN_F19}`\n")
    f.write(f"- output csv: `{OUT_CSV}`\n")
    f.write(f"- rollout rows: `{len(out_rows)}`\n")
    f.write(f"- feature columns: `{len(fields)}`\n\n")

    f.write("## Rows by base condition\n\n")
    f.write("| base_tag | rollouts |\n")
    f.write("|---|---:|\n")
    bc = Counter(r["base_tag"] for r in out_rows)
    for k in sorted(bc):
        f.write(f"| {k} | {bc[k]} |\n")

    f.write("\n## Key derived features\n\n")
    f.write("- `vx_tracking_error_g1`, `vx_tracking_ratio_g1`\n")
    f.write("- `power_per_vx_g1`, `contact_per_vx_g1`, `imu_per_vx_g1`\n")
    f.write("- `lateral_per_forward_g1`\n")
    f.write("- `abs_y_late_minus_early_g1`, `y_late_minus_early_g1`\n")
    f.write("- beta temporal changes: `*_late_minus_early_g1`\n")
    f.write("- context fractions: `context_frac_*`\n\n")

    f.write("## Top absolute feature-target correlations\n\n")
    f.write("| rank | feature | target | corr | abs corr |\n")
    f.write("|---:|---|---|---:|---:|\n")
    for i, (ac, c, fc, tc) in enumerate(corr_rows[:30], 1):
        f.write(f"| {i} | `{fc}` | `{tc}` | {c:.6f} | {ac:.6f} |\n")

    f.write("\n## Safe interpretation\n\n")
    f.write("This table is diagnostic. Some features are rollout-level summaries; before deployment they should be converted into online temporal-window features. Target beta columns are labels and must not be used as input features.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print(f"[TRACER] rollouts={len(out_rows)} cols={len(fields)}")
for r in out_rows[:5]:
    print(f"[TRACER] {r['tag']}: base={r['base_tag']} vx_ratio={r.get('vx_tracking_ratio_g1')} power_per_vx={r.get('power_per_vx_g1')}")
