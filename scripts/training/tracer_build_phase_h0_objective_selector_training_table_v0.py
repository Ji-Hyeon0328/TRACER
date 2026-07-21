#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, pstdev

IN_G1 = Path("datasets/phase_g/g1_enriched_runtime_feature_table_v0.csv")

OUT_CSV = Path("datasets/phase_h/h0_objective_selector_training_table_v0.csv")
OUT_JSON = Path("datasets/phase_h/h0_objective_selector_feature_manifest_v0.json")
OUT_MD = Path("reports/phase_h/h0_objective_selector_training_table_summary_v0.md")

TARGETS = [
    "target_beta_motion_robust_true_metric",
    "target_beta_stability_robust_true_metric",
    "target_beta_energy_robust_true_metric",
]

ID_COLS = [
    "tag",
    "base_tag",
    "n_timestep_rows",
    "manifest",
    "d7_csv",
    "context_mode",
]

CONTEXT_COLS = [
    "context_frac_flat",
    "context_frac_start_flat",
    "context_frac_rough",
    "context_frac_upslope",
    "context_frac_downslope",
    "context_frac_goal_flat",
    "context_frac_unknown",
    "context_unique_count",
]

HAND_PICKED_RUNTIME_FEATURES = [
    # RAM proxy / mismatch proxy
    "ram_slip_proxy_mean_mean_g1",
    "ram_slip_proxy_mean_std_g1",
    "ram_slip_proxy_mean_min_g1",
    "ram_slip_proxy_mean_max_g1",
    "ram_roughness_proxy_mean_mean_g1",
    "ram_roughness_proxy_mean_std_g1",
    "ram_roughness_proxy_mean_min_g1",
    "ram_roughness_proxy_mean_max_g1",
    "ram_sigma_mean_mean_g1",
    "ram_sigma_mean_std_g1",
    "ram_sigma_mean_min_g1",
    "ram_sigma_mean_max_g1",

    # reference summary
    "ref_vx_mean_mean_g1",
    "ref_vx_mean_std_g1",
    "ref_vx_mean_min_g1",
    "ref_vx_mean_max_g1",
    "ref_yaw_rate_mean_mean_g1",
    "ref_yaw_rate_mean_std_g1",
    "ref_yaw_rate_mean_min_g1",
    "ref_yaw_rate_mean_max_g1",
    "ref_clearance_mean_mean_g1",
    "ref_clearance_mean_std_g1",
    "ref_clearance_mean_min_g1",
    "ref_clearance_mean_max_g1",

    # lateral/temporal drift features
    "x_progress_g1",
    "y_change_g1",
    "abs_y_change_g1",
    "y_early_mean_g1",
    "y_mid_mean_g1",
    "y_late_mean_g1",
    "y_late_minus_early_g1",
    "abs_y_early_mean_g1",
    "abs_y_mid_mean_g1",
    "abs_y_late_mean_g1",
    "abs_y_late_minus_early_g1",

    # tracking / contact / energy diagnostic features
    # These are rollout-level now, but should become online temporal-window features later.
    "vx_tracking_error_g1",
    "vx_tracking_abs_error_g1",
    "vx_tracking_ratio_g1",
    "power_per_vx_g1",
    "contact_per_vx_g1",
    "lateral_per_forward_g1",
    "imu_per_vx_g1",
]

DIAGNOSTIC_ONLY_FEATURES = [
    # Artificial reset/lateral offset condition. Useful for current bootstrap diagnostics,
    # not directly deployable as a real robot input.
    "reset_y_mean_g1",
    "reset_y_min_g1",
    "reset_y_max_g1",
]

def ff(x, default=""):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default

def beta_valid(row):
    vals = [ff(row.get(t), "") for t in TARGETS]
    return all(v != "" for v in vals)

def beta_dominant(row):
    vals = [float(row[t]) for t in TARGETS]
    labels = ["motion", "stability", "energy"]
    return labels[max(range(3), key=lambda i: vals[i])]

def feature_available(rows, col):
    if col not in rows[0]:
        return False
    return any(ff(r.get(col), "") != "" for r in rows)

def bad_feature_name(col):
    c = col.lower()

    if c in [x.lower() for x in ID_COLS]:
        return True

    if c in [x.lower() for x in TARGETS]:
        return True

    # Do not train objective selector by copying old D7 beta outputs.
    if c.startswith("pred_beta_") or c.startswith("raw_beta_") or c.startswith("prior_beta_") or c.startswith("actual_beta_") or c.startswith("err_beta_"):
        return True

    # Avoid explicit label/target leakage.
    if c.startswith("target_"):
        return True
    if "true_seed" in c:
        return True
    if "dominant_beta" in c:
        return True

    # Avoid direct true metric columns as selector inputs.
    # Derived online-window proxy features are allowed only if explicitly hand-picked above.
    if c.startswith("rollout_true_metric_"):
        return True
    if c.startswith("robust_group_true_metric_"):
        return True
    if c.startswith("group_"):
        return True

    # These are target-construction scores, not deployable selector inputs.
    if c in ["motion_score", "stability_score", "energy_score", "effort_proxy"]:
        return True
    if c.endswith("_score_mean_g1") or c.endswith("_score_std_g1"):
        return True

    return False

def safe_mean(vals):
    vals = [float(v) for v in vals if v != ""]
    return mean(vals) if vals else ""

def safe_std(vals):
    vals = [float(v) for v in vals if v != ""]
    return pstdev(vals) if len(vals) > 1 else 0.0 if vals else ""

rows = list(csv.DictReader(open(IN_G1)))
if not rows:
    raise SystemExit(f"[TRACER] empty input: {IN_G1}")

rows = [r for r in rows if beta_valid(r)]
if not rows:
    raise SystemExit("[TRACER] no rows with valid beta targets")

context_features = [c for c in CONTEXT_COLS if feature_available(rows, c)]
runtime_features = [c for c in HAND_PICKED_RUNTIME_FEATURES if feature_available(rows, c)]
diagnostic_features = [c for c in DIAGNOSTIC_ONLY_FEATURES if feature_available(rows, c)]

# Extra conservative scan: include only obviously runtime/window-safe columns not already hand-picked.
extra_runtime = []
for col in rows[0].keys():
    if col in context_features or col in runtime_features or col in diagnostic_features:
        continue
    if bad_feature_name(col):
        continue
    if not feature_available(rows, col):
        continue

    lc = col.lower()
    if (
        lc.startswith("context_frac_")
        or lc.startswith("ram_")
        or lc.startswith("ref_")
        or lc.startswith("y_")
        or lc.startswith("abs_y_")
        or lc in ["x_progress_g1", "y_change_g1", "abs_y_change_g1"]
    ):
        extra_runtime.append(col)

extra_runtime = sorted(set(extra_runtime))

deployable_pretrain_features = []
for col in context_features + runtime_features + extra_runtime:
    if col not in deployable_pretrain_features:
        deployable_pretrain_features.append(col)

diagnostic_features = [c for c in diagnostic_features if c not in deployable_pretrain_features]

all_feature_cols = deployable_pretrain_features + diagnostic_features

out_rows = []
for r in rows:
    out = {
        "tag": r.get("tag", ""),
        "base_tag": r.get("base_tag", ""),
        "n_timestep_rows": r.get("n_timestep_rows", ""),
        "beta_motion_target": r.get("target_beta_motion_robust_true_metric", ""),
        "beta_stability_target": r.get("target_beta_stability_robust_true_metric", ""),
        "beta_energy_target": r.get("target_beta_energy_robust_true_metric", ""),
        "beta_dominant_target": beta_dominant(r),
        "is_new_anchor_p045": "1" if r.get("base_tag", "") == "p045" else "0",
    }

    for c in all_feature_cols:
        out[c] = r.get(c, "")

    out_rows.append(out)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "tag",
    "base_tag",
    "n_timestep_rows",
    "beta_motion_target",
    "beta_stability_target",
    "beta_energy_target",
    "beta_dominant_target",
    "is_new_anchor_p045",
] + all_feature_cols

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

manifest = {
    "phase": "H0",
    "purpose": "Objective Selector supervised pretraining table",
    "input": str(IN_G1),
    "output_csv": str(OUT_CSV),
    "n_rows": len(out_rows),
    "n_features_total": len(all_feature_cols),
    "n_features_deployable_pretrain": len(deployable_pretrain_features),
    "n_features_diagnostic_only": len(diagnostic_features),
    "target_columns": {
        "beta_motion_target": "target_beta_motion_robust_true_metric",
        "beta_stability_target": "target_beta_stability_robust_true_metric",
        "beta_energy_target": "target_beta_energy_robust_true_metric",
    },
    "feature_groups": {
        "deployable_pretrain": deployable_pretrain_features,
        "diagnostic_only": diagnostic_features,
    },
    "excluded_policy": [
        "old D7 pred/raw/prior/actual beta outputs excluded",
        "target/true_seed/dominant labels excluded from features",
        "direct rollout_true_metric_* columns excluded except explicitly derived online-window proxy features",
        "reset_y features kept diagnostic-only",
    ],
    "safe_claim": "Targets are robust true-metric teacher seeds for supervised pretraining, not final IRL labels.",
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(manifest, indent=2))

by_base = Counter(r["base_tag"] for r in out_rows)
by_dom = Counter(r["beta_dominant_target"] for r in out_rows)

target_stats = {}
for col in ["beta_motion_target", "beta_stability_target", "beta_energy_target"]:
    vals = [ff(r.get(col), "") for r in out_rows]
    target_stats[col] = {
        "mean": safe_mean(vals),
        "std": safe_std(vals),
        "min": min([float(v) for v in vals if v != ""]) if vals else "",
        "max": max([float(v) for v in vals if v != ""]) if vals else "",
    }

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-H0 Objective Selector Training Table v0\n\n")
    f.write("This builds a rollout-level supervised pretraining table for the Objective Selector beta model.\n\n")
    f.write(f"- input: `{IN_G1}`\n")
    f.write(f"- output csv: `{OUT_CSV}`\n")
    f.write(f"- feature manifest: `{OUT_JSON}`\n")
    f.write(f"- rows: `{len(out_rows)}`\n")
    f.write(f"- total feature columns: `{len(all_feature_cols)}`\n")
    f.write(f"- deployable-pretrain features: `{len(deployable_pretrain_features)}`\n")
    f.write(f"- diagnostic-only features: `{len(diagnostic_features)}`\n\n")

    f.write("## Rows by base condition\n\n")
    f.write("| base_tag | rows |\n")
    f.write("|---|---:|\n")
    for k in sorted(by_base):
        f.write(f"| {k} | {by_base[k]} |\n")

    f.write("\n## Target beta summary\n\n")
    f.write("| target | mean | std | min | max |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for k, st in target_stats.items():
        f.write(
            f"| {k} | {float(st['mean']):.6f} | {float(st['std']):.6f} | "
            f"{float(st['min']):.6f} | {float(st['max']):.6f} |\n"
        )

    f.write("\n## Dominant beta counts\n\n")
    f.write("| dominant target | rows |\n")
    f.write("|---|---:|\n")
    for k in sorted(by_dom):
        f.write(f"| {k} | {by_dom[k]} |\n")

    f.write("\n## Feature policy\n\n")
    f.write("- Old D7 `pred/raw/prior/actual_beta_*` columns are excluded as inputs.\n")
    f.write("- Target beta, true-seed labels, and dominant labels are excluded as inputs.\n")
    f.write("- Direct `rollout_true_metric_*` columns are excluded; derived tracking/contact/energy ratios are kept as diagnostic online-window candidates.\n")
    f.write("- `reset_y_*` features are kept only in the diagnostic-only group because reset offset is an artificial experimental condition.\n\n")

    f.write("## Safe interpretation\n\n")
    f.write("The beta targets are robust true-metric teacher seeds for supervised Objective Selector pretraining. They are not yet final IRL labels. H1 should train both deployable-pretrain and diagnostic variants and report them separately.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_JSON}")
print(f"[TRACER] wrote {OUT_MD}")
print(f"[TRACER] rows={len(out_rows)} features={len(all_feature_cols)} deployable={len(deployable_pretrain_features)} diagnostic={len(diagnostic_features)}")
print("[TRACER] base_tag counts:")
for k in sorted(by_base):
    print(f"  {k}: {by_base[k]}")
print("[TRACER] dominant beta counts:")
for k in sorted(by_dom):
    print(f"  {k}: {by_dom[k]}")
